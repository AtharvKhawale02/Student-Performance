from django.core.management.base import BaseCommand
import pandas as pd

class Command(BaseCommand):
    help = 'Creates an Excel template for student data upload'

    def handle(self, *args, **options):
        # Create sample data with correct column names
        data = {
            'roll_no': ['2024001', '2024002'],
            'name': ['Student 1', 'Student 2'],
            'year_of_study': [1, 2],
            'participation': [85.0, 75.0],
            'assignments': [82.0, 78.0],
            'test_scores': [88.0, 72.0],
            'attendance': [90.0, 85.0],
            'final_grade': [86.0, 76.0]
        }
        
        df = pd.DataFrame(data)
        df.to_excel('student_data_template.xlsx', index=False)
        self.stdout.write(self.style.SUCCESS('Template created successfully'))
