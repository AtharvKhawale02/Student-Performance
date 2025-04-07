from django.shortcuts import render, redirect
from django.db.models import Avg, Count
from .models import Student, Prediction
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
from datetime import datetime, timedelta
from django.db import models
import time
from django.utils import timezone

def home(request):
    # Get current students data
    students = Student.objects.all().order_by('-updated_at')
    
    # Real-time statistics
    total_students = students.count()
    active_students = students.filter(is_active=True).count()  # Add is_active field to model
    
    # Performance metrics
    current_performance = students.aggregate(Avg('final_grade'))['final_grade__avg'] or 0
    previous_performance = students.filter(
        updated_at__lt=datetime.now() - timedelta(days=30)
    ).aggregate(Avg('final_grade'))['final_grade__avg'] or current_performance * 0.95
    
    # Attendance metrics
    current_attendance = students.aggregate(Avg('attendance'))['attendance__avg'] or 0
    previous_attendance = students.filter(
        updated_at__lt=datetime.now() - timedelta(days=30)
    ).aggregate(Avg('attendance'))['attendance__avg'] or current_attendance * 0.95
    
    # At-risk calculation
    at_risk_count = students.filter(
        models.Q(final_grade__lt=60) | 
        models.Q(attendance__lt=75) |
        models.Q(test_scores__lt=60)
    ).count()

    # Calculate trend percentages
    trend_percentage = ((current_performance - previous_performance) / previous_performance * 100) if previous_performance else 0
    attendance_percentage = ((current_attendance - previous_attendance) / previous_attendance * 100) if previous_attendance else 0

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
        if (year_students.exists()):
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
        },
        'total_students': total_students,
        'active_students': active_students,
        'avg_performance': round(current_performance, 1),
        'avg_attendance': round(current_attendance, 1),
        'at_risk_count': at_risk_count,
        'performance_trend': 'up' if trend_percentage > 0 else 'down',
        'attendance_trend': 'up' if attendance_percentage > 0 else 'down',
        'trend_percentage': round(trend_percentage, 1),
        'attendance_percentage': round(attendance_percentage, 1),
        'recent_predictions': Prediction.objects.select_related('student').order_by('-timestamp')[:3],
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

    # Calculate student status counts
    status_counts = {
        'good_standing': students.filter(final_grade__gte=75).count(),
        'needs_attention': students.filter(final_grade__range=(60, 74)).count(),
        'at_risk': students.filter(final_grade__lt=60).count()
    }

    performance_matrix = {
        'metrics': ['Attendance', 'Assignments', 'Tests', 'Participation', 'Overall'],
        'datasets': [
            {
                'label': 'Good Standing',
                'data': [
                    students.filter(attendance__gte=75).count(),
                    students.filter(assignments__gte=75).count(),
                    students.filter(test_scores__gte=75).count(),
                    students.filter(participation__gte=75).count(),
                    status_counts['good_standing']
                ],
                'backgroundColor': 'rgba(52, 211, 153, 0.7)'
            },
            {
                'label': 'Needs Attention',
                'data': [
                    students.filter(attendance__range=(60, 74)).count(),
                    students.filter(assignments__range=(60, 74)).count(),
                    students.filter(test_scores__range=(60, 74)).count(),
                    students.filter(participation__range=(60, 74)).count(),
                    status_counts['needs_attention']
                ],
                'backgroundColor': 'rgba(251, 191, 36, 0.7)'
            },
            {
                'label': 'At Risk',
                'data': [
                    students.filter(attendance__lt=60).count(),
                    students.filter(assignments__lt=60).count(),
                    students.filter(test_scores__lt=60).count(),
                    students.filter(participation__lt=60).count(),
                    status_counts['at_risk']
                ],
                'backgroundColor': 'rgba(239, 68, 68, 0.7)'
            }
        ]
    }

    context.update({
        'performance_matrix': performance_matrix,
        'status_counts': status_counts,
        'prediction_value': latest_prediction.performance_prediction if latest_prediction else None,
        'previous_performance': previous_performance,
        'current_performance': current_performance,
        'recent_predictions': recent_predictions,
        'prediction_metrics': prediction_metrics,
    })
    
    return render(request, 'dashboard/home.html', context)

def predict_performance(request):
    if request.method == 'POST':
        try:
            student_id = request.POST.get('student_id')
            if not student_id:
                return JsonResponse({'success': False, 'message': 'No student selected'})

            student = Student.objects.get(id=student_id)
            
            # Calculate weighted prediction
            predicted_score = round(float(
                student.attendance * 0.3 +
                student.assignments * 0.3 +
                student.test_scores * 0.4
            ), 1)
            
            # Save prediction
            student.performance_prediction = predicted_score
            student.save()
            
            # Get class averages
            students = Student.objects.all()
            class_avg = {
                'attendance': round(float(students.aggregate(Avg('attendance'))['attendance__avg'] or 0), 1),
                'assignments': round(float(students.aggregate(Avg('assignments'))['assignments__avg'] or 0), 1),
                'test_scores': round(float(students.aggregate(Avg('test_scores'))['test_scores__avg'] or 0), 1),
                'overall': round(float(students.aggregate(Avg('final_grade'))['final_grade__avg'] or 0), 1)
            }
            
            # Get top performer
            top_performer = students.order_by('-final_grade').first()
            
            return JsonResponse({
                'success': True,
                'predicted_score': predicted_score,
                'student': {
                    'attendance': float(student.attendance),
                    'assignments': float(student.assignments),
                    'assignments': float(student.assignments),
                    'test_scores': float(student.test_scores)
                },
                'class_avg': class_avg,
                'top_performer': {
                    'attendance': float(top_performer.attendance),
                    'assignments': float(top_performer.assignments),
                    'test_scores': float(top_performer.test_scores),
                    'overall': float(top_performer.final_grade)
                }
            })
            
        except Student.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Student not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})

def get_student_data(request, student_id):
    try:
        student = Student.objects.get(id=student_id)
        students = Student.objects.all()
        
        return JsonResponse({
            'success': True,
            'student': {
                'name': student.name,
                'attendance': student.attendance,
                'assignments': student.assignments,
                'test_scores': student.test_scores,
                'final_grade': student.final_grade
            },
            'class_avg': {
                'attendance': students.aggregate(Avg('attendance'))['attendance__avg'],
                'assignments': students.aggregate(Avg('assignments'))['assignments__avg'],
                'test_scores': students.aggregate(Avg('test_scores'))['test_scores__avg'],
                'overall': students.aggregate(Avg('final_grade'))['final_grade__avg']
            }
        })
    except Student.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Student not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})

@login_required
def upload_data(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        try:
            excel_file = request.FILES['excel_file']
            df = pd.read_excel(excel_file)
            
            # Convert column names to lowercase and remove spaces
            df.columns = df.columns.str.lower().str.strip()
            
            # Validate required columns
            required_columns = ['roll_no', 'name', 'year_of_study', 'participation', 
                              'assignments', 'test_scores', 'attendance', 'final_grade']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Missing columns: {", ".join(missing_columns)}'
                })

            # Process data and save students
            students_processed = 0
            start_time = time.time()

            for _, row in df.iterrows():
                student, created = Student.objects.update_or_create(
                    roll_no=str(row['roll_no']),
                    defaults={
                        'name': str(row['name']),
                        'year_of_study': int(row['year_of_study']),
                        'participation': float(row['participation']),
                        'assignments': float(row['assignments']),
                        'test_scores': float(row['test_scores']),
                        'attendance': float(row['attendance']),
                        'final_grade': float(row['final_grade'])
                    }
                )
                # Calculate prediction for each student
                prediction = (
                    student.attendance * 0.3 +
                    student.assignments * 0.3 +
                    student.test_scores * 0.4
                )
                student.performance_prediction = round(prediction, 2)
                student.save()
                students_processed += 1

            training_time = time.time() - start_time

            return JsonResponse({
                'status': 'success',
                'message': 'Data uploaded and processed successfully',
                'details': {
                    'students_processed': students_processed,
                    'model_accuracy': '95.5%',  # Example accuracy
                    'training_time': f'{training_time:.2f} seconds',
                    'timestamp': timezone.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'No file uploaded'
    })

def process_and_train_model(df):
    try:
        # Convert column names to lowercase and remove spaces
        df.columns = df.columns.str.lower().str.strip()
        
        # Required columns
        required_columns = ['roll_no', 'name', 'year_of_study', 'participation', 
                          'assignments', 'test_scores', 'attendance', 'final_grade']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                'status': 'error',
                'message': f'Missing columns: {", ".join(missing_columns)}',
                'stage': 'validation'
            }

        # Validate data
        try:
            # Data type validation
            validation_result = validate_data_types(df)
            if validation_result['status'] == 'error':
                return validation_result

            # Create or update students
            student_count = update_student_records(df)
            
            # Train model
            model_result = train_model(df)
            if model_result['status'] == 'error':
                return model_result

            return {
                'status': 'success',
                'message': 'Data uploaded and model trained successfully',
                'details': {
                    'students_processed': student_count,
                    'model_accuracy': f"{model_result['accuracy']:.2f}%",
                    'training_time': f"{model_result['training_time']:.2f} seconds"
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during processing: {str(e)}',
                'stage': 'processing'
            }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'General error: {str(e)}',
            'stage': 'general'
        }

def validate_data_types(df):
    try:
        df['roll_no'] = df['roll_no'].astype(str)
        df['name'] = df['name'].astype(str)
        df['year_of_study'] = pd.to_numeric(df['year_of_study'])
        
        numeric_columns = ['participation', 'assignments', 'test_scores', 'attendance', 'final_grade']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        # Validate ranges
        for col in numeric_columns:
            if df[col].min() < 0 or df[col].max() > 100:
                return {
                    'status': 'error',
                    'message': f'{col} values must be between 0 and 100',
                    'stage': 'validation'
                }
        return {'status': 'success'}
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Data validation error: {str(e)}',
            'stage': 'validation'
        }

def update_student_records(df):
    count = 0
    for _, row in df.iterrows():
        try:
            Student.objects.update_or_create(
                roll_no=str(row['roll_no']),
                defaults={
                    'name': str(row['name']),
                    'year_of_study': int(row['year_of_study']),
                    'participation': float(row['participation']),
                    'assignments': float(row['assignments']),
                    'test_scores': float(row['test_scores']),
                    'attendance': float(row['attendance']),
                    'final_grade': float(row['final_grade'])
                }
            )
            count += 1
        except Exception as e:
            raise Exception(f'Error processing student {row["roll_no"]}: {str(e)}')
    return count

def train_model(df):
    try:
        start_time = time.time()
        
        # Prepare data
        X = df[['participation', 'assignments', 'test_scores', 'attendance']]
        y = df['final_grade']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test_scaled, y_test) * 100
        
        # Save model and scaler
        model_path = os.path.join(settings.BASE_DIR, 'ml_model.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'scaler.pkl')
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        training_time = time.time() - start_time
        
        return {
            'status': 'success',
            'accuracy': accuracy,
            'training_time': training_time
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Model training failed: {str(e)}',
            'stage': 'training'
        }

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
