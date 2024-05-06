from django.db import models
from django.utils import timezone
from django.urls import  reverse

# Create your models here.
class Review(models.Model):
    
    #product = models.ForeignKey(Product,related_name='products',on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=200)
    review_comment = models.TextField(max_length=200)
    created = models.DateTimeField(default=timezone.now)
    class Meta:
        ordering = ('-created',)
        
    def __str__(self):
        return self.name
    














    