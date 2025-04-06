from django.db import models
from django.utils import timezone

class Student(models.Model):
    roll_no = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    year_of_study = models.IntegerField(default=1)
    participation = models.FloatField(default=0.0)
    assignments = models.FloatField(default=0.0)
    test_scores = models.FloatField(default=0.0)
    attendance = models.FloatField(default=0.0)
    final_grade = models.FloatField(default=0.0)
    performance_prediction = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_risk_score(self):
        """Calculate risk score based on current metrics"""
        weighted_score = (
            self.attendance * 0.3 +
            self.assignments * 0.3 +
            self.test_scores * 0.4
        )
        return round(weighted_score, 2)

    def save(self, *args, **kwargs):
        if self.performance_prediction is None:
            self.performance_prediction = self.get_risk_score()
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.name} ({self.roll_no})"
