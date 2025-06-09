import os  # Import the os module
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Investor.settings')

application = get_wsgi_application()