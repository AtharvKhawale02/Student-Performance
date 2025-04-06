from django.core.management.base import BaseCommand
from dashboard.models import Student

class Command(BaseCommand):
    help = 'Updates predictions for all students'

    def handle(self, *args, **options):
        students = Student.objects.all()
        updated = 0

        for student in students:
            old_prediction = student.performance_prediction
            new_prediction = student.calculate_prediction()
            
            if old_prediction != new_prediction:
                updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully updated predictions for {updated} students'
            )
        )
