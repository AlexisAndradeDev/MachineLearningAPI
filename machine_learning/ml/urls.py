from django.urls import path

from . import views

urlpatterns = [
    path('models/create', views.MLModelCreate.as_view()),
    path('models/get/<slug:public_id>', views.MLModelGet.as_view())
]

