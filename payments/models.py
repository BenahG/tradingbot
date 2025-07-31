from django.db import models
from users.models import User

class Payment(models.Model):
    PAYMENT_METHODS = (
        ('paypal', 'PayPal'),
        ('mpesa', 'M-Pesa'),
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=10, choices=PAYMENT_METHODS)
    transaction_id = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_completed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username} - {self.amount} {self.payment_method}"