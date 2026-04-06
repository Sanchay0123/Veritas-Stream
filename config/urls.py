from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from apps.forensics.views import upload_and_analyze

urlpatterns = [
    path('admin/', admin.site.urls),  
    path('upload/', upload_and_analyze, name='upload'),
    # Note: If you want the upload page to be your main homepage, uncomment the line below:
    # path('', upload_and_analyze, name='home'),
]

# 🔓 THE FIX: Tell Django to serve user-uploaded media during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)