import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Lead Dev Hack: Check if we are on WSL2 or Windows to set storage
# This will be your 1TB drive symlink location
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'storage')

INSTALLED_APPS = [
    # Django apps...
    'rest_framework',
    'apps.core',
    'apps.analysis',
    'apps.forensics',
]