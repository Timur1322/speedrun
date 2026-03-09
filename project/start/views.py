from django.shortcuts import render

def start(request):
    show_picture = False
    if request.method == "POST":
        show_picture = True
    
    return render(request, "start/start.html", {
        'show_picture': show_picture
    })