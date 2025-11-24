# interviews/views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import subprocess
import re
import json
from pathlib import Path

import whisper
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ---------- OpenAI client ----------
client = OpenAI()

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# ---------- Audio helpers ----------

def extract_audio(video_path, output_path):
    """Extract WAV audio from uploaded video using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",           # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz
        "-ac", "1",      # mono
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def transcribe_audio(audio_path):
    """Run Whisper STT and return ONLY the final text string."""
    model = whisper.load_model("base")   # you can change to "small"/"medium"
    result = model.transcribe(audio_path)
    return result["text"]


# ---------- LLM helpers ----------

def _extract_json_from_text(text):
    """
    Try to pull the first JSON array or object from the model output.
    """
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")
    return json.loads(match.group(1))


def build_conversation_with_llm(transcript_text):
    """
    Use OpenAI to split transcript into turns:
    [{ "speaker": "Interviewer" | "Candidate", "text": "..." }, ...]
    """
    if not transcript_text:
        return []

    prompt = f"""
You are converting an interview transcript into structured dialog.

Rules:
- Split the transcript into turns of speech.
- Each turn must have:
  - "speaker": either "Interviewer" or "Candidate"
  - "text": the spoken text for that turn
- If it's unclear, assume questions are "Interviewer" and answers are "Candidate".
- Return ONLY valid JSON: a list of objects, no extra commentary.

Transcript:
\"\"\"{transcript_text}\"\"\""""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You label interview dialogs as Interviewer and Candidate."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        data = _extract_json_from_text(raw)
        # Ensure it's a list of {speaker, text}
        conv = []
        for item in data:
            speaker = item.get("speaker", "Speaker")
            text = item.get("text", "").strip()
            if text:
                conv.append({"speaker": speaker, "text": text})
        if conv:
            return conv
    except Exception:
        pass

    # Fallback: single-speaker transcript
    return [{"speaker": "Speaker", "text": transcript_text}]


def extract_qa_pairs_with_llm(transcript_text):
    """
    Use OpenAI to extract Q&A pairs from the transcript.
    Returns: [{ "question": "...", "answer": "..." }, ...]
    """
    if not transcript_text:
        return []

    prompt = f"""
From the following interview transcript, extract clear question-answer pairs.

Rules:
- "question" should be what the interviewer asks.
- "answer" should be the candidate's response to that question.
- If multiple back-and-forth sentences form one answer, merge them.
- Skip greetings / chit-chat that are not real questions.
- Return ONLY valid JSON: a list of objects with "question" and "answer".

Transcript:
\"\"\"{transcript_text}\"\"\""""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You extract structured Q&A from interviews."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        data = _extract_json_from_text(raw)
        qa_list = []
        for item in data:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if q and a:
                qa_list.append({"question": q, "answer": a})
        return qa_list
    except Exception:
        return []


def save_qa_to_output(qa_list, video_name):
    """
    Save Q&A pairs as JSON in output/ folder.
    File name: qa_<video_stem>.json
    """
    if not qa_list:
        return None
    stem = Path(video_name).stem
    out_path = OUTPUT_DIR / f"qa_{stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    return str(out_path)


# ---------- View ----------

def interview_view(request):
    context = {}

    if request.method == "POST":
        video = request.FILES.get("interview_video")
        if video:
            # ---- SAVE VIDEO ----
            fs = FileSystemStorage(location="media/interviews/")
            video_name = fs.save(video.name, video)
            video_path = os.path.join("media/interviews/", video_name)

            # ---- AUDIO SEPARATION ----
            base, _ = os.path.splitext(video_path)
            audio_path = base + ".wav"
            extract_audio(video_path, audio_path)

            # ---- SPEECH TO TEXT (WHISPER) ----
            transcript_text = transcribe_audio(audio_path)

            # ---- LLM: build conversation with Interviewer / Candidate ----
            conversation = build_conversation_with_llm(transcript_text)

            # ---- LLM: extract Q&A pairs & save to output/ ----
            qa_pairs = extract_qa_pairs_with_llm(transcript_text)
            qa_file_path = save_qa_to_output(qa_pairs, video_name)

            # ---- CONTEXT FOR TEMPLATE ----
            context.update(
                {
                    "msg": f"Video '{video_name}' uploaded successfully!",
                    "audio_file": audio_path,
                    "transcript": transcript_text,
                    "conversation": conversation,       # [{speaker, text}, ...]
                    # optional: show where Q&A is saved
                    "qa_file_path": qa_file_path,
                }
            )

    return render(request, "interview.html", context)
