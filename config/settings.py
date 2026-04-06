import environ
import os
from pathlib import Path

# Initialize environ
env = environ.Env(
    DEBUG=(bool, False)
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Read the .env file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# ==========================================
# 🛡️ SECURITY & CORE SETTINGS
# ==========================================
SECRET_KEY = env('SECRET_KEY')
DEBUG = env('DEBUG')
RSA_KEY_PATH = os.path.expanduser('~/.veritas_keys/private_key.pem')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'


# ==========================================
# 💾 DATABASE
# ==========================================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# ==========================================
# 🧩 APPLICATION DEFINITION
# ==========================================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # 🧠 Veritas Stream Custom Apps
    'apps.forensics',
    'apps.analysis',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [], 
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


# ==========================================
# 📁 STATIC & MEDIA STORAGE (The Fix)
# ==========================================
STATIC_URL = 'static/'

# The base URL that serves the media files to the browser
# Matches the <img src="/media/quarantine/..."> request in your HTML
MEDIA_URL = '/media/'

# The absolute path to the directory where files are saved on your hard drive
# Routes Django directly to: <project_root>/apps/storage/
MEDIA_ROOT = os.path.join(BASE_DIR, 'apps', 'storage')

# Keep your vault root for your background engine pipeline
VAULT_ROOT = env('VAULT_STORAGE_PATH')