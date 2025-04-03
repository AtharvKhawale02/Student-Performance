from django.urls import path
from . import views

app_name = 'authentication'

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),  # Ensure this points to login_view
    path('logout/', views.logout_view, name='logout'),
    path('', views.login_view, name='root_redirect'),  # Redirect root URL to login page
    path('base/', views.base_view, name='base'),  # Add path for base.html
]
