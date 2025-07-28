from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from users.forms import UserRegistrationForm

def index(request):
    """Render the main index page."""
    context = {
        'page_title': 'Home Page',
        'active_page': 'home'
    }
    return render(request, 'index.html', context)

@require_http_methods(["GET", "POST"])
def admin_login(request):
    """Handle admin login page."""
    context = {
        'page_title': 'Admin Login',
        'active_page': 'admin_login'
    }
    return render(request, 'admin_login.html', context)

@require_http_methods(["GET", "POST"])
def user_login(request):
    """Handle user login page."""
    context = {
        'page_title': 'User Login',
        'active_page': 'user_login'
    }
    return render(request, 'user_login.html', context)

@require_http_methods(["GET", "POST"])
def user_register(request):
    """
    Handle user registration with form validation.
    """
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Registration successful! Please log in.')
            return redirect('user_login')
    else:
        form = UserRegistrationForm()

    context = {
        'form': form,
        'page_title': 'User Registration',
        'active_page': 'register'
    }
    return render(request, 'user_registration.html', context)
