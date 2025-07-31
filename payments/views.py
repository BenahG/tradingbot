from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.urls import reverse
from paypal.standard.forms import PayPalPaymentsForm
from .models import Payment
from users.models import User
from datetime import datetime, timedelta
import requests
from django.utils import timezone
import base64
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@login_required
def payment_options(request):
    return render(request, 'payments/options.html')

@login_required
def process_paypal(request):
    # Create payment record
    payment = Payment.objects.create(
        user=request.user,
        amount=settings.SUBSCRIPTION_PRICE,
        payment_method='paypal',
        transaction_id='pending',
        is_completed=False
    )
    
    # PayPal integration
    paypal_dict = {
        "business": settings.PAYPAL_CLIENT_ID,
        "amount": str(settings.SUBSCRIPTION_PRICE),
        "item_name": "Weekly Trading Bot Subscription",
        "invoice": f"tradingbot-{payment.id}",
        "currency_code": "USD",
        "notify_url": request.build_absolute_uri(reverse('paypal-ipn')),
        "return_url": request.build_absolute_uri(reverse('payment_success')),
        "cancel_return": request.build_absolute_uri(reverse('payment_cancel')),
    }
    
    form = PayPalPaymentsForm(initial=paypal_dict)
    return render(request, "payments/process_paypal.html", {"form": form})

@login_required
def process_mpesa(request):
    if request.method == 'POST':
        phone = request.POST.get('phone')
        
        # M-Pesa STK Push implementation
        access_token = get_mpesa_access_token()
        if not access_token:
            return render(request, 'payments/mpesa_error.html')
            
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        password = base64.b64encode((settings.MPESA_SHORTCODE + settings.MPESA_PASSKEY + timestamp).encode()).decode()
        
        payload = {
            "BusinessShortCode": settings.MPESA_SHORTCODE,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": str(int(settings.SUBSCRIPTION_PRICE)),
            "PartyA": phone,
            "PartyB": settings.MPESA_SHORTCODE,
            "PhoneNumber": phone,
            "CallBackURL": request.build_absolute_uri(reverse('mpesa_callback')),
            "AccountReference": "TRADINGBOT",
            "TransactionDesc": "Weekly Subscription"
        }
        
        response = requests.post(
            "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return render(request, 'payments/mpesa_pending.html')
        else:
            return render(request, 'payments/mpesa_error.html')
    
    return render(request, 'payments/process_mpesa.html')

def get_mpesa_access_token():
    auth = (settings.MPESA_CONSUMER_KEY, settings.MPESA_CONSUMER_SECRET)
    response = requests.get(
        "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials",
        auth=auth
    )
    if response.status_code == 200:
        return response.json().get('access_token')
    return None

@login_required
def payment_success(request):
    # Update user subscription
    user = request.user
    user.is_premium = True
    if user.premium_expiry and user.premium_expiry > timezone.now():
        user.premium_expiry += timedelta(days=settings.SUBSCRIPTION_PERIOD)
    else:
        user.premium_expiry = timezone.now() + timedelta(days=settings.SUBSCRIPTION_PERIOD)
    user.save()
    
    # Update payment record
    payment = Payment.objects.filter(user=user, is_completed=False).last()
    if payment:
        payment.transaction_id = request.GET.get('tx', 'unknown')
        payment.is_completed = True
        payment.save()
    
    return render(request, 'payments/success.html')

@login_required
def payment_cancel(request):
    return render(request, 'payments/cancel.html')

@csrf_exempt
def mpesa_callback(request):
    """
    This view will be called by Safaricom after the user completes STK push.
    You can log or store response for further validation.
    """
    if request.method == 'POST':
        # Log or process M-PESA callback data
        import json
        data = json.loads(request.body)
        print("M-PESA Callback Data:", data)  # For debugging

        # TODO: Add validation and update payment records

        return JsonResponse({"ResultCode": 0, "ResultDesc": "Accepted"})
    return JsonResponse({"ResultCode": 1, "ResultDesc": "Invalid request method"}, status=400)