from django.shortcuts import render

def start(request: HttpRequest) -> HttpResponse:
    context = {
        'button_clicked': False


    if request.method == "POST":
        context['button_clicked'] = True


    return render(request, "start.html", {"form": form})
