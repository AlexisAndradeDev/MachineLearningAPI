# Generated by Django 4.0.6 on 2022-07-13 23:51

import django.core.validators
from django.db import migrations, models
import ml.models


class Migration(migrations.Migration):

    dependencies = [
        ('ml', '0002_rename_current_time_mlmodel_create_time_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='DataSet',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50, validators=[django.core.validators.MinLengthValidator(3)])),
                ('public_id', models.SlugField(unique=True)),
                ('file', models.FileField(upload_to=ml.models.dataset_file_path)),
                ('create_time', models.DateField(auto_now_add=True)),
                ('last_modified', models.DateField(auto_now=True)),
            ],
        ),
    ]