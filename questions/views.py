from django.shortcuts import render

def generate_questions(request):
    questions = []
    if request.method == 'POST':
        domain = request.POST.get('domain')
        # TODO: integrate your LLM/RAG question generation logic
        questions = [f"Sample {domain} Question {i+1}" for i in range(5)]
    return render(request, 'question_gen.html', {'questions': questions})
