from django.urls import path
from .views import FaceVerifyView

app_name = "api"

urlpatterns = [
    path('verify/', FaceVerifyView.as_view(), name='face-verify'),
]