from django.db import models
from django.utils import timezone

class Student(models.Model):
    roll_no = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    year_of_study = models.IntegerField()
    participation = models.FloatField(default=0)
    assignments = models.FloatField(default=0)
    test_scores = models.FloatField(default=0)
    attendance = models.FloatField(default=0)
    final_grade = models.FloatField(null=True, blank=True)
    performance_prediction = models.FloatField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def calculate_final_grade(self):
        """Calculate final grade using weighted components"""
        weights = {
            'test_scores': 0.4,
            'assignments': 0.3,
            'participation': 0.2,
            'attendance': 0.1
        }
        
        self.final_grade = (
            self.test_scores * weights['test_scores'] +
            self.assignments * weights['assignments'] +
            self.participation * weights['participation'] +
            self.attendance * weights['attendance']
        )
        return self.final_grade

    def get_risk_score(self):
        """Calculate risk score based on current performance metrics"""
        # Different weights for risk assessment
        weights = {
            'attendance': 0.35,
            'test_scores': 0.35,
            'assignments': 0.2,
            'participation': 0.1
        }
        
        return (
            self.attendance * weights['attendance'] +
            self.test_scores * weights['test_scores'] +
            self.assignments * weights['assignments'] +
            self.participation * weights['participation']
        )

    def get_risk_status(self):
        """Return risk status and color based on score"""
        score = self.get_risk_score()
        if score >= 75:
            return 'Good Standing', 'green'
        elif score >= 50:
            return 'Needs Attention', 'yellow'
        else:
            return 'At Risk', 'red'

    def save(self, *args, **kwargs):
        if not self.final_grade:
            self.calculate_final_grade()
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['roll_no']

    def __str__(self):
        return f"{self.roll_no} - {self.name}"
