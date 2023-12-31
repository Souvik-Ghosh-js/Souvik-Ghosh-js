# Generated by Django 4.1.7 on 2023-05-14 21:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_rename_username_profile_username_profile_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='MyModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Ethnicity', models.IntegerField()),
                ('Gender', models.IntegerField()),
                ('AgeGroup', models.CharField(max_length=200)),
                ('BMI', models.FloatField(max_length=200)),
                ('Hydration', models.FloatField()),
            ],
            options={
                'db_table': 'hyd',
            },
        ),
    ]
