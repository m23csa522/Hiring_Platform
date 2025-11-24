from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import subprocess
import torch
# do not import heavy/optional libs at module import time (can break Django startup)

def extract_audio(video_path, output_path):
    """Extract WAV audio from uploaded video using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",           # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz
        "-ac", "1",      # mono
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def transcribe_audio(audio_path):
    """Whisper STT."""
    try:
        import whisper
    except Exception as e:
        raise RuntimeError(
            "Could not import `whisper`. Install the correct package (e.g. 'openai-whisper') "
            f"or ensure the environment is configured. Import error: {e}"
        )

    model = whisper.load_model("base")   # small/medium/base
    result = model.transcribe(audio_path)
    return result["text"]

def speaker_identification(audio_path):
    """Simple speaker diarization using pyannote."""
    from pyannote.audio import Pipeline
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        # You can choose to raise or just skip diarization gracefully
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Please set it before running speaker identification."
        )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append(f"{speaker}: {turn.start:.1f}s → {turn.end:.1f}s")
    return speakers


def interview_view(request):
    context = {}

    if request.method == "POST":
        video = request.FILES.get("interview_video")
        if video:
            fs = FileSystemStorage(location="media/interviews/")
            video_name = fs.save(video.name, video)
            video_path = os.path.join("media/interviews/", video_name)

            # ---- AUDIO SEPARATION ----
            audio_path = video_path.replace(".mp4", ".wav")
            extract_audio(video_path, audio_path)

            # ---- SPEAKER IDENTIFICATION ----
            try:
                speakers = speaker_identification(audio_path)
            except Exception as e:
                speakers = [f"Speaker diarization failed: {e}"]

            # ---- SPEECH TO TEXT ----
            try:
                transcript = transcribe_audio(audio_path)
            except Exception as e:
                transcript = f"Transcription failed: {e}"

            fs = FileSystemStorage(location="media/transcripts/")
            filename = fs.save(video_name.replace(".mp4", ".txt"), transcript)

            context.update({
                "msg": f"Video '{video_name}' uploaded successfully!",
                "audio_file": audio_path,
                "transcript": transcript,
                "speakers": speakers
            })

    return render(request, "interview.html", context)
