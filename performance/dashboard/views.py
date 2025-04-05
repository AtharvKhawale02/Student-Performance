from django.shortcuts import render, redirect
from django.db.models import Avg, Count
from .models import Student
import joblib
import os
from django.conf import settings
from django.http import JsonResponse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from django.contrib.auth.decorators import login_required
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def home(request):
    students = Student.objects.all().order_by('-updated_at')
    
    # Calculate statistics
    avg_performance = students.aggregate(Avg('final_grade'))['final_grade__avg']
    avg_attendance = students.aggregate(Avg('attendance'))['attendance__avg']
    
    # Calculate at-risk count
    at_risk_count = sum(1 for student in students if student.get_risk_score() < 50)
    
    # Calculate trends (comparing to previous month)
    current_performance = avg_performance if avg_performance else 0
    current_attendance = avg_attendance if avg_attendance else 0
    
    # You would typically get these from historical data
    # For now using static comparison
    previous_performance = current_performance * 0.95  # Example: 5% lower last month
    previous_attendance = current_attendance * 0.90    # Example: 10% lower last month
    
    # Calculate individual vs average performance metrics
    avg_metrics = {
        'participation': students.aggregate(Avg('participation'))['participation__avg'] or 0,
        'assignments': students.aggregate(Avg('assignments'))['assignments__avg'] or 0,
        'test_scores': students.aggregate(Avg('test_scores'))['test_scores__avg'] or 0,
        'attendance': students.aggregate(Avg('attendance'))['attendance__avg'] or 0
    }
    
    # Get performance by year
    performance_by_year = {}
    for year in range(1, 5):
        year_avg = students.filter(year_of_study=year).aggregate(
            Avg('final_grade'))['final_grade__avg']
        performance_by_year[year] = year_avg or 0
    
    # Calculate grade distribution
    grade_distribution = {
        'A': students.filter(final_grade__gte=90).count(),
        'B': students.filter(final_grade__range=(80, 89)).count(),
        'C': students.filter(final_grade__range=(70, 79)).count(),
        'D': students.filter(final_grade__range=(60, 69)).count(),
        'F': students.filter(final_grade__lt=60).count()
    }
    
    # Calculate performance metrics over time
    performance_trends = {
        'assignments': list(students.values_list('assignments', flat=True)),
        'test_scores': list(students.values_list('test_scores', flat=True)),
        'attendance': list(students.values_list('attendance', flat=True)),
        'participation': list(students.values_list('participation', flat=True))
    }

    # Calculate performance matrix data
    performance_matrix = {
        'metrics': ['Assignments', 'Tests', 'Attendance', 'Participation'],
        'current': [
            avg_metrics['assignments'],
            avg_metrics['test_scores'],
            avg_metrics['attendance'],
            avg_metrics['participation']
        ],
        'target': [85, 80, 90, 85]  # Target percentages for each metric
    }

    # Enhanced year-wise performance calculation
    year_performance = {}
    for year in range(1, 5):
        year_students = students.filter(year_of_study=year)
        if year_students.exists():
            year_performance[year] = {
                'avg_grade': year_students.aggregate(Avg('final_grade'))['final_grade__avg'] or 0,
                'count': year_students.count(),
                'passing': year_students.filter(final_grade__gte=60).count()
            }
        else:
            year_performance[year] = {'avg_grade': 0, 'count': 0, 'passing': 0}

    context = {
        'students': students,
        'avg_performance': current_performance,
        'avg_attendance': current_attendance,
        'at_risk_count': at_risk_count,
        'performance_trend': 'up' if current_performance > previous_performance else 'down',
        'attendance_trend': 'up' if current_attendance > previous_attendance else 'down',
        'trend_percentage': round(((current_performance - previous_performance) / previous_performance) * 100 if previous_performance else 0, 1),
        'attendance_percentage': round(((current_attendance - previous_attendance) / previous_attendance) * 100 if previous_attendance else 0, 1),
        'avg_metrics': avg_metrics,
        'performance_by_year': performance_by_year,
        'student_names': [student.name for student in students],
        'student_performances': [student.final_grade for student in students],
        'grade_distribution': grade_distribution,
        'performance_trends': performance_trends,
        'chart_data': {
            'weekly_performance': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                'datasets': {
                    'assignments': list(students.values_list('assignments', flat=True)),
                    'test_scores': list(students.values_list('test_scores', flat=True)),
                    'attendance': list(students.values_list('attendance', flat=True))
                }
            }
        },
        'performance_matrix': performance_matrix,
        'year_performance': year_performance,
        'yearly_data': {
            'labels': ['1st Year', '2nd Year', '3rd Year', '4th Year'],
            'values': [
                year_performance[1]['avg_grade'],
                year_performance[2]['avg_grade'],
                year_performance[3]['avg_grade'],
                year_performance[4]['avg_grade']
            ],
            'counts': [
                year_performance[1]['count'],
                year_performance[2]['count'],
                year_performance[3]['count'],
                year_performance[4]['count']
            ]
        }
    }
    
    # Add prediction value for the chart
    latest_prediction = None
    if students.exists():
        latest_prediction = students.filter(
            performance_prediction__isnull=False
        ).order_by('-id').first()

    # Get recent predictions (last 5) with timestamp handling
    recent_predictions = []
    for student in students.filter(performance_prediction__isnull=False)[:5]:
        recent_predictions.append({
            'student': {
                'name': student.name,
                'id': student.id
            },
            'score': student.performance_prediction,
            'timestamp': student.updated_at,
            'actual': student.final_grade
        })

    # Calculate prediction accuracy metrics
    prediction_metrics = {
        'previous_month': previous_performance,
        'current': current_performance,
        'predicted': latest_prediction.performance_prediction if latest_prediction else 0,
        'accuracy': 95.5,  # Example accuracy score - you should calculate this based on your model
        'improvement': 2.3  # Example improvement - calculate based on historical data
    }

    context.update({
        'prediction_value': latest_prediction.performance_prediction if latest_prediction else None,
        'previous_performance': previous_performance,
        'current_performance': current_performance,
        'recent_predictions': recent_predictions,
        'prediction_metrics': prediction_metrics,
    })
    
    return render(request, 'dashboard/home.html', context)

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

@login_required
def upload_data(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        try:
            excel_file = request.FILES['excel_file']
            
            # Read Excel file
            df = pd.read_excel(excel_file)
            
            # Create or update students
            for _, row in df.iterrows():
                Student.objects.update_or_create(
                    roll_no=row['roll_no'],
                    defaults={
                        'name': row['name'],
                        'year_of_study': int(row['year_of_study']),
                        'participation': float(row['participation']),
                        'assignments': float(row['assignments']),
                        'test_scores': float(row['test_scores']),
                        'attendance': float(row['attendance']),
                        'final_grade': float(row['final_grade'])
                    }
                )
            
            # Prepare data for training
            X = df[['participation', 'assignments', 'test_scores', 'attendance']]
            y = df['final_grade']
            
            # Split data and scale features
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train a more sophisticated model
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            model_path = os.path.join(settings.BASE_DIR, 'ml_model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'scaler.pkl')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Update predictions using scaled features
            students = Student.objects.all()
            for student in students:
                features = scaler.transform([
                    [
                        student.participation,
                        student.assignments,
                        student.test_scores,
                        student.attendance
                    ]
                ])
                prediction = model.predict(features)
                student.performance_prediction = prediction[0]
                student.save()
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return redirect('dashboard:home')

@login_required
def get_realtime_metrics(request):
    """API endpoint for real-time metrics"""
    students = Student.objects.all()
    
    data = {
        'metrics': {
            'assignments': list(students.values_list('assignments', flat=True)),
            'test_scores': list(students.values_list('test_scores', flat=True)),
            'attendance': list(students.values_list('attendance', flat=True))
        }
    }
    return JsonResponse(data)
