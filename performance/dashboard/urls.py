from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_performance, name='predict'),
    path('upload-data/', views.upload_data, name='upload_data'),
    path('metrics/', views.get_realtime_metrics, name='metrics'),
]
