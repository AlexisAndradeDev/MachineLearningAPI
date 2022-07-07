from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import MLModel
from .serializers import MLModelSerializer

# Create your views here.

class MLModelCreate(generics.CreateAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

class MLModelGet(generics.RetrieveAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    lookup_field = 'public_id'
