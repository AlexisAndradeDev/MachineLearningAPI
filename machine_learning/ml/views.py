from rest_framework import generics

from .models import MLModel
from .serializers import MLModelSerializer

# Create your views here.

class MLModelCreateAPIView(generics.CreateAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

class MLModelGetAPIView(generics.RetrieveAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    lookup_field = 'public_id'
