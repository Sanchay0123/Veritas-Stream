import os
from django.conf import settings
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

print(f"Key Path: {settings.RSA_KEY_PATH}")
if os.path.exists(settings.RSA_KEY_PATH):
    print("✅ Key is accessible and secure.")
else:
    print("❌ Cannot find the key. Check your .env path.")