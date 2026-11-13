from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User

# UserAdmin gives a proper password-change form, grouped fields, search and filters.
admin.site.register(User, UserAdmin)
