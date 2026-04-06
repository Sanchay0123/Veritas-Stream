import environ
import os
from pathlib import Path

# Initialize environ with a strict default for production safety
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
# Pulls from .env, but defaults to False if missing (Critical for Prod!)
DEBUG = env('DEBUG') 
RSA_KEY_PATH = os.path.expanduser('~/.veritas_keys/private_key.pem')
VT_API_KEY = env('VT_API_KEY')  # VirusTotal API Key from .env

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'

# 🚀 SUBDOMAIN CONFIGURATION
# We allow the tunnel address and the local loopback
ALLOWED_HOSTS = [
    'veritas-stream.sanchayjain.in', 
    'localhost', 
    '127.0.0.1',
    '[::1]'
]

# Required for Django's CSRF protection to work over the HTTPS tunnel
CSRF_TRUSTED_ORIGINS = ['https://veritas-stream.sanchayjain.in']


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
# 📁 STATIC & MEDIA STORAGE
# ==========================================
# The base URL for static files
STATIC_URL = '/static/'
# 🛑 NEW: Where collectstatic will dump your assets for Cloudflare
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# The base URL that serves the media files to the browser
MEDIA_URL = '/media/'
# The absolute path routing directly to your WSL2 storage architecture
MEDIA_ROOT = os.path.join(BASE_DIR, 'apps', 'storage')

# Keep your vault root for your background engine pipeline
VAULT_ROOT = env('VAULT_STORAGE_PATH')