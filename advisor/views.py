from django.shortcuts import render, redirect
from django.views import View
from .models import UserProfile, InvestmentSuggestion
from .forms import UserProfileForm

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