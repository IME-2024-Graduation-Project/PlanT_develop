from django.contrib import admin
from .models import Place, Route, Trip, Plan, Tag, User


# Register your models here.
admin.site.register(Place)
admin.site.register(Route)
admin.site.register(Trip)
admin.site.register(Plan)
admin.site.register(Tag)
admin.site.register(User)