"""WSGI config for RAIM."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raim.settings")

application = get_wsgi_application()
