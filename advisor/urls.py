from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse
from . import views

# Simple homepage view
def homepage(request):
    return HttpResponse("""
        <h1>Welcome to AI Investor</h1>
        <p>Go to <a href='/advisor/profile/'>Profile</a></p>
        <p>Go to <a href='/integrated/'>Integrated Function</a></p>
    """)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('advisor/', include('advisor.urls')),  # Include advisor app's URLs
    path('', homepage, name='homepage'),  # Root URL
    path('integrated/', views.integrated_view, name='integrated'),
]