from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    phone_number = forms.CharField(required=False)
    address = forms.CharField(required=False, widget=forms.Textarea)

    class Meta:
        model = CustomUser
        fields = ("username", "email", "first_name", "last_name", "phone_number", "address", "password1", "password2")

class CustomAuthenticationForm(AuthenticationForm):
    class Meta:
        model = CustomUser
