from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100)
    attendance = models.FloatField()
    test_scores = models.FloatField()
    assignments = models.FloatField()
    performance_prediction = models.FloatField(null=True, blank=True)
