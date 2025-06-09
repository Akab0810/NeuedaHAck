from django.test import TestCase
from .models import UserProfile, InvestmentSuggestion

class UserProfileModelTest(TestCase):
    def setUp(self):
        self.user_profile = UserProfile.objects.create(
            user='testuser',
            age=30,
            income=70000,
            risk_tolerance='medium',
            investment_goals='retirement'
        )

    def test_user_profile_creation(self):
        self.assertEqual(self.user_profile.user, 'testuser')
        self.assertEqual(self.user_profile.age, 30)
        self.assertEqual(self.user_profile.income, 70000)
        self.assertEqual(self.user_profile.risk_tolerance, 'medium')
        self.assertEqual(self.user_profile.investment_goals, 'retirement')

class InvestmentSuggestionModelTest(TestCase):
    def setUp(self):
        self.user_profile = UserProfile.objects.create(
            user='testuser',
            age=30,
            income=70000,
            risk_tolerance='medium',
            investment_goals='retirement'
        )
        self.investment_suggestion = InvestmentSuggestion.objects.create(
            user=self.user_profile,
            suggestion_text='Invest in index funds',
        )

    def test_investment_suggestion_creation(self):
        self.assertEqual(self.investment_suggestion.user, self.user_profile)
        self.assertEqual(self.investment_suggestion.suggestion_text, 'Invest in index funds')