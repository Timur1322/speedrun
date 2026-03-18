from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import UserProfile
import os
import json
import joblib
import numpy as np
import tensorflow as tf
import hashlib as h
from collections import Counter
from django.conf import settings
from django.http import JsonResponse
from .models import UploadResult


MODEL = tf.keras.models.load_model('model2.h5')
ENCODER = joblib.load('label_encoder.pkl') 
def login_view(request):
    msg = ''
    if request.user.is_authenticated:
        if request.user.userprofile.role == 'admin':
            return redirect('admin_page')
        return redirect('user_page')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            if user.userprofile.role == 'admin':
                return redirect('admin_page')
            return redirect('user_page')
        else:
            msg = 'Неверный логин или пароль'

    return render(request, 'start/login.html', {'msg': msg})


def logout_view(request):
    logout(request)
    return redirect('login')



def admin_page(request):


    msg = ''
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')

        if User.objects.filter(username=username).exists():
            msg = 'Такой пользователь уже есть'
        else:
            user = User.objects.create_user(
                username=username,
                password=password
            )
            UserProfile.objects.create(
                user=user,
                first_name=first_name,
                last_name=last_name,
                role='user'
            )
            msg = f'Пользователь {username} создан'

    return render(request, 'start/admin_page.html', {'msg': msg})


@login_required
def profile_page(request):
    return render(request, 'start/profile.html', {
        'profile': request.user.userprofile
    })

def clean_labels(labels):
    result = []
    for x in labels:
        x = x.decode() if isinstance(x, bytes) else str(x)
        result.append(1 if len(x) > 32 else x)
    return np.array(result)


@login_required
def user_page(request):
    if request.user.userprofile.role != 'user':
        return redirect('start/admin_page')

    history = {
        'accuracy': [0.5, 0.62, 0.7, 0.78, 0.84, 0.88],
        'val_accuracy': [0.48, 0.6, 0.68, 0.75, 0.8, 0.85]
    }

    class_counts = {
        '0': 120,
        '1': 115,
        '2': 130,
        '3': 108,
        '4': 122
    }

    uploads = UploadResult.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'start/user_page.html', {
        'history': json.dumps(history),
        'class_counts': json.dumps(class_counts),
        'uploads': uploads
    })



def get_races_from_labels(labels_data):
    """
    восстановления числовых меток (0-49) из строк вида Hash(Salt(race_id)+name)+Name
    """
    races = []
    for y in labels_data:
        hi = y[:32]
        name = y[32:]
        found = False
        for race_id in range(50):
            test_hash = h.md5((str(race_id) + name).encode()).hexdigest()
            if test_hash == hi:
                races.append(race_id)
                found = True
                break
        if not found:
            races.append(0) 
    return np.array(races)

def predict_view(request):
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.role != 'user':
        return JsonResponse({'error': 'Нет доступа'}, status=403)

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        
        try:
            # Загружаем данные из .npz
            data = np.load(file, allow_pickle=True)
            test_x = data['valid_x']
            test_y_raw = data.get('valid_y', None)

            # Если x пришел как (N, 80000), превращаем в (N, 80000, 1)
            if len(test_x.shape) == 2:
                test_x = np.expand_dims(test_x, axis=-1)

            # 4. Предсказание
            predictions_raw = MODEL.predict(test_x, verbose=0)
            pred_indices = np.argmax(predictions_raw, axis=1)
            # Переводим индексы обратно в названия (если нужно) или оставляем ID
            pred_labels = ENCODER.inverse_transform(pred_indices)

            top_5_counts = dict(Counter(pred_labels.tolist()).most_common(5))

            accuracy = 0
            loss = 0
            per_sample_results = []

            if test_y_raw is not None:
                # Восстанавливаем числовые метки из хэшей
                true_indices = get_races_from_labels(test_y_raw)
                
                loss, accuracy = MODEL.evaluate(test_x, true_indices, verbose=0)
                
                # Сравнение по каждому образцу (1 - угадали, 0 - нет)
                per_sample_results = (pred_indices == true_indices).astype(int).tolist()

            UploadResult.objects.create(
                user=request.user,
                file_name=file.name,
                accuracy=float(accuracy),
                loss=float(loss)
            )

            return JsonResponse({
                'status': 'success',
                'accuracy': round(float(accuracy), 4),
                'loss': round(float(loss), 4),
                'top_5': top_5_counts,
                'per_sample': per_sample_results
            })

        except Exception as e:
            return JsonResponse({'error': f'Ошибка обработки: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Файл не загружен или метод не POST'}, status=400)