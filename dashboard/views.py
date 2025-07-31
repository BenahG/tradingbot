from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import BotConfiguration
from users.models import User
from django.contrib import messages
import json
from bot.learnpattern import MarketUnderstandingBot
from django.http import JsonResponse
from threading import Thread
import time

# Store active bot instances
active_bots = {}

@login_required
def dashboard(request):
    if not request.user.has_active_subscription():
        messages.warning(request, "You need an active subscription to access the trading bot")
        return redirect('payment_options')
    
    try:
        config = BotConfiguration.objects.get(user=request.user)
    except BotConfiguration.DoesNotExist:
        config = None
    
    symbol_choices = [
        ('GOLD', 'Gold'),
        ('XAUUSD', 'XAU/USD'),
        ('BTCUSD', 'BTC/USD'),
        ('US100Cash', 'US100 Cash'),
    ]
    
    context = {
        'config': config,
        'symbol_choices': symbol_choices,
        'bot_active': request.user.username in active_bots
    }
    return render(request, 'dashboard/dashboard.html', context)

@login_required
def save_config(request):
    if request.method == 'POST':
        symbols = request.POST.getlist('symbols')
        config_data = {
            'account': request.POST.get('account'),
            'password': request.POST.get('password'),
            'server': request.POST.get('server'),
            'target_profit_usd': float(request.POST.get('target_profit_usd', 1000)),
            'risk_percent': float(request.POST.get('risk_percent', 1)),
            'symbols': symbols,
            'max_trades_per_symbol': int(request.POST.get('max_trades_per_symbol', 1)),
            'is_active': False
        }
        
        config, created = BotConfiguration.objects.update_or_create(
            user=request.user,
            defaults=config_data
        )
        
        messages.success(request, "Configuration saved successfully")
        return redirect('dashboard')
    
    return redirect('dashboard')

@login_required
def start_bot(request):
    if not request.user.has_active_subscription():
        return JsonResponse({'status': 'error', 'message': 'Subscription required'})
    
    try:
        config = BotConfiguration.objects.get(user=request.user)
    except BotConfiguration.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Configuration missing'})
    
    if request.user.username in active_bots:
        return JsonResponse({'status': 'error', 'message': 'Bot already running'})
    
    # Create bot configuration dictionary
    bot_config = {
        'account': config.account,
        'password': config.password,
        'server': config.server,
        'target_profit_usd': float(config.target_profit_usd),
        'risk_percent': float(config.risk_percent),
        'symbols': config.symbols,
        'max_trades_per_symbol': int(config.max_trades_per_symbol),
    }
    
    # Start bot in a separate thread
    def run_bot():
        bot = MarketUnderstandingBot()
        bot.config.update(bot_config)
        active_bots[request.user.username] = bot
        bot.run()
        del active_bots[request.user.username]
    
    thread = Thread(target=run_bot)
    thread.daemon = True
    thread.start()
    
    # Update config to active
    config.is_active = True
    config.save()
    
    return JsonResponse({'status': 'success', 'message': 'Bot started successfully'})

@login_required
def stop_bot(request):
    if request.user.username not in active_bots:
        return JsonResponse({'status': 'error', 'message': 'No active bot found'})
    
    bot = active_bots[request.user.username]
    bot.close_all_trades()
    
    # This will cause the bot to exit its run loop
    del active_bots[request.user.username]
    
    # Update config
    try:
        config = BotConfiguration.objects.get(user=request.user)
        config.is_active = False
        config.save()
    except BotConfiguration.DoesNotExist:
        pass
    
    return JsonResponse({'status': 'success', 'message': 'Bot stopped successfully'})

@login_required
def bot_status(request):
    if request.user.username in active_bots:
        bot = active_bots[request.user.username]
        status = {
            'active': True,
            'symbols': bot.config['symbols'],
            'active_trades': list(bot.active_trades.keys()),
            'total_profit': bot.total_profit,
            'target_profit': bot.config['target_profit_usd']
        }
    else:
        status = {'active': False}
    
    return JsonResponse(status)