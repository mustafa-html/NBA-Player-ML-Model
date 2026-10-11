"""Production entry point: gunicorn wsgi:app"""
from webapp import create_app

app = create_app()
