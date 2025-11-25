from django.shortcuts import render
import requests

def generate_questions(request):
    questions = []
    if request.method == 'POST':
        domain = request.POST.get('domain')

        url = "http://127.0.0.1:7001/generate_questions"
        payload = {
            "num_questions": 5,
            "topic": domain,
            "resume_id": request.session.get("resume_id") if request.session.get("resume_id") else "Bhavesh_Wadhwani_Resume.pdf",
            "jd_id": request.session.get("jd_id") if request.session.get("jd_id") else "JD-1.pdf",
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response = response.json()
            questions = response.get("questions", [])

    return render(request, 'question_gen.html', {'questions': questions})