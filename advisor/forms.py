# filepath: c:\Course\Neueda Hackathon\AI_Investor\advisor\forms.py
from django import forms
from .models import UserProfile

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['age', 'income', 'risk_tolerance', 'investment_goals']