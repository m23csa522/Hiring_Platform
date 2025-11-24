from django.shortcuts import render

def reports_view(request):
    data = {'candidate': 'John Doe', 'technical': 85, 'communication': 90, 'confidence': 80}
    return render(request, 'reports.html', {'data': data})
