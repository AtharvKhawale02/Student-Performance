from django.contrib import admin
from .models import Student

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('roll_no', 'name', 'year_of_study', 'final_grade', 'is_active')
    list_filter = ('year_of_study', 'is_active')
    search_fields = ('roll_no', 'name')
    ordering = ('-updated_at',)
