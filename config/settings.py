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

# --- Security Settings ---
SECRET_KEY = env('SECRET_KEY')
DEBUG = env('DEBUG')
RSA_KEY_PATH = env('RSA_PRIVATE_KEY_PATH')

# --- Storage Settings ---
MEDIA_ROOT = os.path.join(BASE_DIR, 'storage')
VAULT_ROOT = env('VAULT_STORAGE_PATH')


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Your Veritas Apps
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
        'DIRS': [], # Add strings like os.path.join(BASE_DIR, 'templates') later
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


# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Internal project URLs
ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'


# The absolute path to the directory where files are saved
MEDIA_ROOT = os.path.join(BASE_DIR, 'storage')

# The URL that serves the media files (for development only)
MEDIA_URL = '/media/'