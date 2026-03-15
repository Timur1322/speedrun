# import numpy as np
# import tensorflow as tf
# import joblib, json
# from django.shortcuts import render, redirect
# from django.http import JsonResponse
# from .models import UserProfile
# from django.contrib.auth import authenticate, login

# MODEL = tf.keras.models.load_model('model.h5')
# LE = joblib.load('label_encoder.pkl')

# def user_dashboard(request):
#     if not request.user.is_authenticated: return redirect('login')
    
#     with open('history.json', 'r') as f:
#         history_data = json.load(f)
        
#     return render(request, 'user_dashboard.html', {'history': json.dumps(history_data)})

# def predict_signals(request):
#     if request.method == 'POST' and request.FILES.get('file'):
#         npz_file = np.load(request.FILES['file'])
#         test_x = npz_file['test_x']
        
#         # Работа ИИ
#         predictions = MODEL.predict(test_x)
#         class_indices = np.argmax(predictions, axis=1)
#         class_names = LE.inverse_transform(class_indices)
        
#         # Считаем точность
#         accuracy = 0
#         if 'test_y' in npz_file:
#             # Очистка меток от хеша
#             true_labels = [n[32:] for n in npz_file['test_y']]
#             accuracy = np.mean(class_names == true_labels)

#         # Подготовка данных для топ-5 часто встречающихся классов
#         from collections import Counter
#         counts = Counter(class_names)
#         top_5 = dict(counts.most_common(5))

#         return JsonResponse({
#             'accuracy': float(accuracy),
#             'top_5': top_5,
#             'all_counts': dict(counts)
#         })
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from .models import UserProfile

def create_user_view(request):
    # Проверка на админа (роль проверяем из профиля)
    if not request.user.is_authenticated or request.user.userprofile.role != 'admin':
        return redirect('login')

    if request.method == 'POST':
        u = request.POST['username']
        p = request.POST['password']
        fn = request.POST['first_name']
        ln = request.POST['last_name']


        new_user = User.objects.create_user(username=u, password=p)
        
        UserProfile.objects.create(user=new_user, first_name=fn, last_name=ln, role='user')
        
        return render(request, 'admin_panel.html', {'msg': f'Пользователь {u} создан!'})
    
    return render(request, 'create_user.html')