from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    RISK_TOLERANCE_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.PositiveIntegerField()
    income = models.DecimalField(max_digits=10, decimal_places=2)
    risk_tolerance = models.CharField(max_length=6, choices=RISK_TOLERANCE_CHOICES)
    investment_goals = models.TextField()

    def __str__(self):
        return self.user.username

class InvestmentSuggestion(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    suggestion_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Suggestion for {self.user.user.username} on {self.created_at}'