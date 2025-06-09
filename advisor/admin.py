from django.contrib import admin
from .models import UserProfile, InvestmentSuggestion

admin.site.register(UserProfile)
admin.site.register(InvestmentSuggestion)