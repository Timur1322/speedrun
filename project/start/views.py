from django.shortcuts import render, redirect

def start(request):
    if request.method == "POST":

        current = request.GET.get('show', '0')
        new_value = '0' if current == '1' else '1'
        return redirect(f'/start/?show={new_value}')
    
  
    show_picture = request.GET.get('show', '0') == '1'
    
    return render(request, "start/start.html", {
        'show_picture': show_picture
    })