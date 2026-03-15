import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from django.contrib.auth.models import User
from start.models import UserProfile

if not User.objects.filter(username='admin').exists():
    user = User.objects.create_user(username='admin', password='admin123')
    UserProfile.objects.create(
        user=user,
        first_name='Главный',
        last_name='Админ',
        role='admin'adsaf
    )
    print('admin created')
else:
    print('admin exists')