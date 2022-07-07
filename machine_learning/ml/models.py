from time import time

from django.db import models
from django.core.validators import MinLengthValidator, MinValueValidator
from django.utils.text import slugify

# Create your models here.

type_of_model_by_algorithm = {
    "regression": "supervised",
    "classification": "supervised",
}

class MLModel(models.Model):
    name = models.CharField(max_length=50, validators=[MinLengthValidator(3)])
    public_id = models.SlugField(unique=True)
    algorithm = models.CharField(max_length=50)
    cost_function = models.CharField(null=True, max_length=50)
    epochs = models.IntegerField(validators=[MinValueValidator(1)])
    lr = models.FloatField(default=0.001)

    current_time = models.DateField(auto_now_add=True)
    last_modified = models.DateField(auto_now=True)

    @property
    def type_of_model(self):
        """Supervised / unsupervised"""
        return type_of_model_by_algorithm[self.algorithm]

    def __str__(self):
        return f'{self.name} - {self.public_id}'

    def generate_unique_id(self):
        """
        Generates a unique ID using the current time and the name of the
        model.

        Returns:
            str: Unique ID.
        """        
        strtime = ''.join(str(time()).split('.'))
        unique_id = f'{strtime[7:]}-{self.name}'
        unique_id = slugify(unique_id)
        return unique_id

    def save(self, *args, **kwargs):
        self.public_id = self.generate_unique_id()
        super(MLModel, self).save()
    