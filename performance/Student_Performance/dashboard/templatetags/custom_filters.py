from django import template

register = template.Library()

@register.filter
def average(queryset, field_name):
    """
    Calculate the average of a specific field in a queryset.
    """
    values = [getattr(obj, field_name, 0) for obj in queryset if getattr(obj, field_name, None) is not None]
    if values:
        return sum(values) / len(values)
    return 0
