from django.shortcuts import render
import requests
from django.conf import settings
from pathlib import Path

def reports_view(request):
    data = None
    average_score = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "run_eval":
            media_root = Path(settings.MEDIA_ROOT) if getattr(settings, 'MEDIA_ROOT', None) else Path(settings.BASE_DIR) / 'media'
            transcripts_dir = media_root / 'transcripts'
            transcripts = [p.name for p in transcripts_dir.glob('*.txt')] if transcripts_dir.exists() else []

            if transcripts:
                first_path = transcripts_dir / transcripts[0]
                try:
                    with first_path.open('r', encoding='utf-8') as f:
                        first_transcript = f.read()
                except Exception:
                    first_transcript = ''
            else:
                first_transcript = ''

            if len(first_transcript) > 0:
                url = "http://127.0.0.1:7001/extract-transcript"
                payload = {'transcript_text': first_transcript}
                headers = {'Content-Type': 'application/json'}
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    qapairs = response.json().get('qa_pairs', [])
                else:
                    qapairs = []
            else:
                qapairs = []

            if len(qapairs) > 0:
                evaluation_result = []   
                url = "http://127.0.0.1:7001/evaluate"
                for i in qapairs:
                    payload = {"question": i.get("question"), "user_answer": i.get("answer")}
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        evaluation = response.json()
                        evaluation_result.append(evaluation)
                    else:
                        # Handle 
                        print("Evaluation request failed for question:", i.get("question"))
                        pass
            else:
                evaluation_result = []

            if len(evaluation_result) > 0:
                average_score = sum(item.get('score', 0) for item in evaluation_result) / len(evaluation_result)
                data = evaluation_result
            else:
                data = None
                average_score = None

    return render(request, 'reports.html', {'data': data, 'average_score': average_score})
