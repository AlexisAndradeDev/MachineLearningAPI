from django.db import models
from django.core.validators import MinLengthValidator, MinValueValidator

# Create your models here.

type_of_model_by_algorithm = {
    "regression": "supervised",
    "classification": "supervised",
}

class MLModel(models.Model):
    name = models.CharField(max_length=50, validators=[MinLengthValidator(3)])
    algorithm = models.CharField(max_length=50)
    cost_function = models.CharField(null=True, max_length=50)
    epochs = models.IntegerField(validators=[MinValueValidator(1)])
    lr = models.FloatField(default=0.001)

    @property
    def type_of_model(self):
        """Supervised / unsupervised"""
        return type_of_model_by_algorithm[self.algorithm]
