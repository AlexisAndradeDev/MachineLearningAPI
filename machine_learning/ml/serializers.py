from rest_framework import serializers

from .models import MLModel

class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = [
            'name',
            'public_id',
            'func',
            'cost_function',
            'create_time',
            'last_modified',
        ]

