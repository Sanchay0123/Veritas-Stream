from django.apps import AppConfig

class ForensicsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.forensics'  # The full path
    label = 'forensics'      # The short name for migrations