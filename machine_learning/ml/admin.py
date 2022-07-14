from django.contrib import admin
from .models import MLModel, DataSet

# Register your models here.

class DataSetAdmin(admin.ModelAdmin):
    exclude = ('public_id',)

admin.site.register(MLModel)
admin.site.register(DataSet, DataSetAdmin)
