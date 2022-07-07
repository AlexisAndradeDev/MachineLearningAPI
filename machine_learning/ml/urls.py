from django.urls import path

from . import views

urlpatterns = [
    path('models/create', views.MLModelCreateAPIView.as_view()),
    path('models/get/<slug:public_id>', views.MLModelGetAPIView.as_view())
]

