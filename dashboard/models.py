from django.db import models
from users.models import User

class BotConfiguration(models.Model):
    SYMBOL_CHOICES = [
        ('GOLD', 'Gold'),
        ('XAUUSD', 'XAU/USD'),
        ('BTCUSD', 'BTC/USD'),
        ('US100Cash', 'US100 Cash'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    account = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    server = models.CharField(max_length=100)
    target_profit_usd = models.DecimalField(max_digits=10, decimal_places=2, default=1000)
    risk_percent = models.DecimalField(max_digits=5, decimal_places=2, default=1)
    symbols = models.JSONField(default=list)  # Stores list of symbols
    max_trades_per_symbol = models.IntegerField(default=1)
    is_active = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username}'s Bot Configuration"