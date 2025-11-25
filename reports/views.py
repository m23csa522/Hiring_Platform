from django.shortcuts import render
import requests
from django.conf import settings
from pathlib import Path

def reports_view(request):
    data = None
    average_score = None

    if request.method == "POST":
        action = request.POST.get("action")

        # ---------------------------
        # RUN EVALUATION
        # ---------------------------
        if action == "run_eval":
            media_root = Path(settings.BASE_DIR) / 'output'
            qa_pairs_path = media_root / 'qa_interview_4JKCana.json'

            try:
                with qa_pairs_path.open('r', encoding='utf-8') as f:
                    import json
                    qapairs = json.load(f)
            except:
                qapairs = []

            evaluation_result = []
            if qapairs:
                url = "http://127.0.0.1:7001/evaluate"

                for i in qapairs:
                    payload = {
                        "question": i.get("question"),
                        "user_answer": i.get("answer")
                    }
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        evaluation_result.append(response.json())

            if evaluation_result:
                average_score = sum(i.get("score", 0) for i in evaluation_result) / len(evaluation_result)
                data = evaluation_result
            else:
                data = None
                average_score = None

            request.session["data"] = data
            request.session["average_score"] = average_score

            return render(request, "reports.html", {
                "data": data,
                "average_score": average_score
            })

        # ---------------------------
        # DOWNLOAD PDF
        # ---------------------------
        if action == "download_pdf":
            from xhtml2pdf import pisa
            from django.http import HttpResponse
            from django.template.loader import render_to_string

            # Get stored session data
            data = request.session.get("data")
            average_score = request.session.get("average_score")

            # Use a dedicated PDF template
            html = render_to_string("reports_pdf.html", {
                "data": data,
                "average_score": average_score
            })

            response = HttpResponse(content_type="application/pdf")
            response["Content-Disposition"] = "attachment; filename=report.pdf"

            pisa_status = pisa.CreatePDF(html, dest=response)

            if pisa_status.err:
                return HttpResponse("Error generating PDF", status=500)

            return response

    return render(request, "reports.html", {
        "data": data,
        "average_score": average_score
    })

