from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    is_premium = models.BooleanField(default=False)
    premium_expiry = models.DateTimeField(null=True, blank=True)
    broker_account = models.CharField(max_length=100, blank=True)
    broker_password = models.CharField(max_length=100, blank=True)
    broker_server = models.CharField(max_length=100, blank=True)
    
    def has_active_subscription(self):
        if self.is_premium and self.premium_expiry:
            from django.utils import timezone
            return self.premium_expiry > timezone.now()
        return False