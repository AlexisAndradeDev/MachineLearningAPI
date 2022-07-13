from time import time

from django.db import models
from django.core.validators import MinLengthValidator, MinValueValidator
from django.utils.text import slugify

# Create your models here.

class MLModel(models.Model):
    name = models.CharField(max_length=50, validators=[MinLengthValidator(3)])
    public_id = models.SlugField(unique=True)
    func = models.CharField(max_length=50) # linear, sigmoid, etc.
    cost_function = models.CharField(null=True, max_length=50) # mse, binary-crossentropy, etc.

    create_time = models.DateField(auto_now_add=True)
    last_modified = models.DateField(auto_now=True)

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
    