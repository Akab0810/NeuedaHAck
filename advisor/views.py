from django.shortcuts import render, redirect
from django.views import View
from .models import UserProfile, InvestmentSuggestion
from .forms import UserProfileForm
import sys
import os

# Import integrated.py (adjust path if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integrated  # This imports integrated.py from the project root

class UserProfileView(View):
    def get(self, request):
        form = UserProfileForm()
        return render(request, 'profile_form.html', {'form': form})

    def post(self, request):
        form = UserProfileForm(request.POST)
        if form.is_valid():
            user_profile = form.save(commit=False)
            user_profile.user = request.user
            user_profile.save()
            return redirect('suggestions_list')
        return render(request, 'profile_form.html', {'form': form})

class InvestmentSuggestionView(View):
    def get(self, request):
        suggestions = InvestmentSuggestion.objects.filter(user=request.user)
        return render(request, 'suggestions_list.html', {'suggestions': suggestions})

def integrated_view(request):
    result = None
    if request.method == "POST":
        # Call your function from integrated.py
        result = integrated.run_integration()  # Replace with actual function
    return render(request, "advisor/integrated.html", {"result": result})