from django.contrib import admin
from django.urls import path
from apps.forensics.views import upload_and_analyze

urlpatterns = [
    # Change .path to .urls
    path('admin/', admin.site.urls),  
    path('upload/', upload_and_analyze, name='upload'),
]