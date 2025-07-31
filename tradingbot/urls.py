from django.contrib import admin
from django.urls import path, include
from users import views as user_views
from payments import views as payment_views
from dashboard import views as dashboard_views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('register/', user_views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path('profile/', user_views.profile, name='profile'),
    path('accounts/', include('django.contrib.auth.urls')),
    
    # Payment URLs
    path('payments/', payment_views.payment_options, name='payment_options'),
    path('payments/paypal/', payment_views.process_paypal, name='process_paypal'),
    path('payments/mpesa/', payment_views.process_mpesa, name='process_mpesa'),
    path('payments/success/', payment_views.payment_success, name='payment_success'),
    path('payments/cancel/', payment_views.payment_cancel, name='payment_cancel'),
    path('payments/mpesa/callback/', payment_views.mpesa_callback, name='mpesa_callback'),
    path('paypal/', include('paypal.standard.ipn.urls')),
    
    # Dashboard URLs
    path('', dashboard_views.dashboard, name='dashboard'),
    path('save_config/', dashboard_views.save_config, name='save_config'),
    path('start_bot/', dashboard_views.start_bot, name='start_bot'),
    path('stop_bot/', dashboard_views.stop_bot, name='stop_bot'),
    path('bot_status/', dashboard_views.bot_status, name='bot_status'),

    # Password change URLs
    path('password-change/', 
         auth_views.PasswordChangeView.as_view(template_name='users/password_change.html'), 
         name='password_change'),
    path('password-change/done/', 
         auth_views.PasswordChangeDoneView.as_view(template_name='users/password_change_done.html'), 
         name='password_change_done'),
]