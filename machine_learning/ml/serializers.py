from rest_framework import serializers

from .models import MLModel

class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = [
            'name',
            'algorithm',
            'cost_function',
            'epochs',
            'lr',
            'type_of_model',
        ]

