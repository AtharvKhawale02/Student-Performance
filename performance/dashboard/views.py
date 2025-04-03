from django.shortcuts import render, redirect
from .models import Student
import joblib
import os
from django.conf import settings

def home(request):
    # Fetch all students from the database
    students = Student.objects.all()
    return render(request, 'dashboard/home.html', {'students': students})

def predict_performance(request):
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        if not student_id:
            return redirect('dashboard:home')  # Redirect if no student_id is provided
        try:
            student = Student.objects.get(id=student_id)
        except Student.DoesNotExist:
            return redirect('dashboard:home')  # Redirect if student does not exist

        # Load the ML model
        model_path = os.path.join(settings.BASE_DIR, 'performance', 'ml_model.pkl')
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            return redirect('dashboard:home')  # Redirect if model file is missing

        # Predict performance
        prediction = model.predict([[student.attendance, student.test_scores, student.assignments]])
        student.performance_prediction = prediction[0]
        student.save()
    return redirect('dashboard:home')
