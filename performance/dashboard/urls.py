from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),  # Map root URL of dashboard to the home view
    path('predict/', views.predict_performance, name='predict'),  # Path for performance prediction
]
