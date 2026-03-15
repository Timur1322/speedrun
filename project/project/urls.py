from start import views
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('start/', views.start),
    path('', include('start.urls'))
]
