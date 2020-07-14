from django.db import models
from .storage import OverwriteStorage
# Create your models here.
class file(models.Model):
    file_name = models.CharField(max_length=200)
    Content = models.FileField(storage=OverwriteStorage())
    def __str__(self):
        return self.file_name
