# Generated by Django 3.2.14 on 2022-07-07 22:45

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MLModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50, validators=[django.core.validators.MinLengthValidator(3)])),
                ('public_id', models.SlugField(unique=True)),
                ('algorithm', models.CharField(max_length=50)),
                ('cost_function', models.CharField(max_length=50, null=True)),
                ('epochs', models.IntegerField(validators=[django.core.validators.MinValueValidator(1)])),
                ('lr', models.FloatField(default=0.001)),
                ('current_time', models.DateField(auto_now_add=True)),
                ('last_modified', models.DateField(auto_now=True)),
            ],
        ),
    ]
