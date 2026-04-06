from django.contrib import admin
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve
from apps.forensics.views import upload_and_analyze

urlpatterns = [
    path('admin/', admin.site.urls),  
    path('upload/', upload_and_analyze, name='upload'),
    path('', upload_and_analyze, name='home'), 
]

# 🔓 THE FIX: Force Django to serve static and media files through the tunnel
# This explicitly bypasses the DEBUG=False restriction
urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT,
    }),
    re_path(r'^static/(?P<path>.*)$', serve, {
        'document_root': settings.STATIC_ROOT,
    }),
]