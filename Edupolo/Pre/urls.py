from django.urls import path
from . import views


urlpatterns = [
    path('saludo/', views.respu, name='saludo'),
]
