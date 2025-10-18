from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.views.decorators.http import require_http_methods
from users.forms import UserRegistrationForm

@require_http_methods(["GET"])
def index(request):
    """Render the main landing page with context."""
    return render(request, 'index.html', {
        'page_title': 'Welcome',
        'active_page': 'home'
    })

@require_http_methods(["GET", "POST"])
def admin_login(request):
    """Handle admin authentication securely."""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            messages.success(request, 'Admin login successful!')
            return redirect('admin_dashboard')
        messages.error(request, 'Invalid admin credentials')
    
    return render(request, 'admin/login.html', {
        'page_title': 'Admin Login',
        'active_page': 'admin_login'
    })

@require_http_methods(["GET", "POST"])
def user_login(request):
    """Handle user authentication."""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, 'Login successful!')
            return redirect('user_dashboard')
        messages.error(request, 'Invalid credentials')
    
    return render(request, 'accounts/login.html', {
        'page_title': 'User Login',
        'active_page': 'user_login'
    })

@require_http_methods(["GET", "POST"])
def user_register(request):
    """Handle user registration with validation."""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Registration successful! Please login.')
            return redirect('user_login')
    else:
        form = UserRegistrationForm()

    return render(request, 'accounts/registration.html', {
        'form': form,
        'page_title': 'User Registration',
        'active_page': 'register'
    })
