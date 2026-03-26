import hashlib as h
import numpy as np
import json
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from start.models import UserProfile, UploadResult
from start.views import clean_labels, get_races_from_labels


class StartViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()

        self.regular_user = User.objects.create_user(
            username='testuser',
            password='password123'
        )
        UserProfile.objects.create(user=self.regular_user, role='user')

        self.admin_user = User.objects.create_superuser(
    username='adminuser',
    password='password123',
    email='admin@example.com')
        UserProfile.objects.create(user=self.admin_user, role='admin')

        self.login_url = reverse('login')
        self.logout_url = reverse('logout')
        self.admin_page_url = reverse('admin_page')
        self.user_page_url = reverse('user_page')
        self.profile_url = reverse('profile')
        self.predict_url = reverse('predict')
    def test_login_get(self):
        """Проверяем, что GET на страницу логина работает корректно."""
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'start/login.html')
        self.assertTrue(self.client.login(username='testuser', password='password123'))

    def test_login_post_success_user(self):
        """Успешный вход обычного пользователя."""
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'password123'
        })
        self.assertRedirects(response, self.user_page_url)

    def test_login_post_success_admin(self):
        """Успешный вход администратора."""
        response = self.client.post(self.login_url, {
            'username': 'adminuser',
            'password': 'password123'
        })
        self.assertRedirects(response, self.admin_page_url)

    def test_login_post_failure(self):
        """Неуспешный вход — неверный пароль."""
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['msg'], 'Неверный логин или пароль')

    def test_logout_view(self):
        """Проверяем, что logout перенаправляет на логин."""
        self.client.login(username='testuser', password='password123')
        response = self.client.get(self.logout_url)
        self.assertRedirects(response, self.login_url)

    def test_admin_page_create_user(self):
        """Создание нового пользователя через админ‑страницу."""
        response = self.client.post(self.login_url, {
            'username': 'adminuser',
            'password': 'password123'
        })
        response = self.client.post(self.admin_page_url, {
            'username': 'newuser',
            'password': 'newpassword',
            'first_name': 'New',
            'last_name': 'User'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['msg'], 'Пользователь newuser создан')
        self.assertTrue(User.objects.filter(username='newuser').exists())


    def test_user_page_as_admin_redirects(self):
        """Администратор не должен попадать на user_page."""
        self.client.login(username='adminuser', password='password123')
        response = self.client.get(self.user_page_url)
        self.assertEqual(response.status_code, 403)

 
    def test_clean_labels(self):
        """Проверяем обработку строк, байтов и длинных значений (>32)."""
        labels = [b'short_bytes', 'short_str', 'a' * 33]
        result = clean_labels(labels)
        expected = np.array(['short_bytes', 'short_str', '1'])
        np.testing.assert_array_equal(result, expected)

    def test_get_races_from_labels(self):
        """Проверяем корректное извлечение race_id из хеша."""
        race_id = 5
        name = "John"
        hashed = h.md5((str(race_id) + name).encode()).hexdigest()
        label = hashed + name

        result = get_races_from_labels([label])
        self.assertEqual(result, [5])


    def test_predict_view_forbidden(self):
        """Предсказание запрещено без авторизации или если роль не user."""

        self.client.login(username='adminuser', password='password123')
        response = self.client.post(self.predict_url)
        self.assertEqual(response.status_code, 403)

   