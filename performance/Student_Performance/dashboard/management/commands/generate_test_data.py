from django.core.management.base import BaseCommand
from dashboard.models import Student
import random

class Command(BaseCommand):
    help = 'Generate test student data'

    def handle(self, *args, **kwargs):
        # Clear existing data
        Student.objects.all().delete()

        # Generate 20 test students
        for i in range(20):
            attendance = random.uniform(60.0, 100.0)
            assignments = random.uniform(60.0, 100.0)
            test_scores = random.uniform(60.0, 100.0)
            participation = random.uniform(60.0, 100.0)
            final_grade = (attendance * 0.3 + 
                         assignments * 0.3 + 
                         test_scores * 0.4)

            Student.objects.create(
                roll_no=f"2024{str(i+1).zfill(3)}",
                name=f"Student {i+1}",
                year_of_study=random.randint(1, 4),
                participation=round(participation, 2),
                assignments=round(assignments, 2),
                test_scores=round(test_scores, 2),
                attendance=round(attendance, 2),
                final_grade=round(final_grade, 2)
            )
        
        self.stdout.write(self.style.SUCCESS('Successfully generated test data'))
