from django.shortcuts import render, redirect
from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_http_methods
from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel

def is_admin(user):
    """Check if user is admin."""
    return user.is_authenticated and user.is_staff

@require_http_methods(["GET", "POST"])
def admin_login(request):
    """Handle admin authentication securely."""
    if request.method == 'POST':
        username = request.POST.get('loginid')
        password = request.POST.get('pswd')
        user = auth.authenticate(username=username, password=password)
        
        if user is not None and user.is_staff:
            auth.login(request, user)
            messages.success(request, 'Logged in successfully!')
            return redirect('admin_home')
        else:
            messages.error(request, 'Invalid credentials or not an admin')
    
    return render(request, 'admins/admin_login.html', {
        'page_title': 'Admin Login'
    })

@login_required
@user_passes_test(is_admin)
def admin_home(request):
    """Admin dashboard view."""
    return render(request, 'admins/admin_home.html', {
        'page_title': 'Admin Dashboard'
    })

@login_required
@user_passes_test(is_admin)
def register_users_view(request):
    """View all registered users with pagination."""
    users = UserRegistrationModel.objects.all().order_by('-created_at')
    return render(request, 'admins/view_register_users.html', {
        'users': users,
        'page_title': 'Registered Users'
    })

@login_required
@user_passes_test(is_admin)
@require_http_methods(["POST"])
def activate_users(request):
    """Activate/deactivate users securely."""
    user_id = request.POST.get('uid')
    try:
        user = UserRegistrationModel.objects.get(id=user_id)
        new_status = 'activated' if user.status != 'activated' else 'pending'
        user.status = new_status
        user.save()
        messages.success(request, f'User {user.username} status updated to {new_status}')
    except UserRegistrationModel.DoesNotExist:
        messages.error(request, 'User not found')
    
    return redirect('register_users_view')
