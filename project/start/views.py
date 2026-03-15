from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import UserProfile


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
    return render(request, 'profile.html', {
        'profile': request.user.userprofile
    })
import os
import json
import joblib
import numpy as np
import tensorflow as tf
from collections import Counter
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import UploadResult

MODEL = ''#tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'model.h5'))
ENCODER = ''#joblib.load(os.path.join(settings.BASE_DIR, 'label_encoder.pkl'))


def clean_labels(labels):
    result = []
    for x in labels:
        x = x.decode() if isinstance(x, bytes) else str(x)
        result.append(x[32:] if len(x) > 32 else x)
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

    return render(request, 'user_page.html', {
        'history': json.dumps(history),
        'class_counts': json.dumps(class_counts),
        'uploads': uploads
    })


@login_required
def predict_view(request):
    if request.user.userprofile.role != 'user':
        return JsonResponse({'error': 'Нет доступа'}, status=403)

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        data = np.load(file, allow_pickle=True)

        test_x = data['test_x']
        test_y = data['test_y'] if 'test_y' in data else None

        if len(test_x.shape) == 2:
            test_x = np.expand_dims(test_x, axis=-1)

        pred = MODEL.predict(test_x, verbose=0)
        pred_index = np.argmax(pred, axis=1)
        pred_labels = ENCODER.inverse_transform(pred_index)

        top = dict(Counter(pred_labels).most_common(5))

        accuracy = 0
        loss = 0
        per_sample = []

        if test_y is not None:
            true_labels = clean_labels(test_y)
            true_index = ENCODER.transform(true_labels)
            true_cat = tf.keras.utils.to_categorical(true_index, num_classes=len(ENCODER.classes_))
            loss, accuracy = MODEL.evaluate(test_x, true_cat, verbose=0)
            per_sample = (pred_labels == true_labels).astype(int).tolist()

        UploadResult.objects.create(
            user=request.user,
            file_name=file.name,
            accuracy=float(accuracy),
            loss=float(loss)
        )

        return JsonResponse({
            'accuracy': float(accuracy),
            'loss': float(loss),
            'top_5': top,
            'per_sample': per_sample
        })

    return JsonResponse({'error': 'Файл не загружен'}, status=400)
