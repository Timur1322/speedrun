from django.urls import path
from .views import *
urlpatterns = [
    path('', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('admin-page/', admin_page, name='admin_page'),
    path('profile/', profile_page, name='profile'),
    path('user-page/', user_page, name='user_page'),
    path('predict/', predict_view, name='predict'),
]