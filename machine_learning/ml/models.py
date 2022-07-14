from django.db import models
from django.core.validators import MinLengthValidator, MinValueValidator

from modules.models_tools import generate_unique_id

# Create your models here.

def dataset_file_path(instance, filename):
    return f'datasets/{instance.public_id}.csv'

class DataSet(models.Model):
    name = models.CharField(max_length=50, validators=[MinLengthValidator(3)])
    public_id = models.SlugField(unique=True)
    file = models.FileField(upload_to=dataset_file_path)
    create_time = models.DateField(auto_now_add=True)
    last_modified = models.DateField(auto_now=True)

    def __str__(self):
        return f'{self.name} - {self.public_id}'

    def save(self, *args, **kwargs):
        if not self.public_id:
            self.public_id = generate_unique_id(self.name)
        super(DataSet, self).save()

    def delete(self):
        self.file.delete()
        super(DataSet, self).delete()

class MLModel(models.Model):
    name = models.CharField(max_length=50, validators=[MinLengthValidator(3)])
    public_id = models.SlugField(unique=True)
    func = models.CharField(max_length=50) # linear, sigmoid, etc.
    cost_function = models.CharField(null=True, max_length=50) # mse, binary-crossentropy, etc.

    create_time = models.DateField(auto_now_add=True)
    last_modified = models.DateField(auto_now=True)

    def __str__(self):
        return f'{self.name} - {self.public_id}'

    def save(self, *args, **kwargs):
        if not self.public_id:
            self.public_id = generate_unique_id(self.name)
        super(MLModel, self).save()
