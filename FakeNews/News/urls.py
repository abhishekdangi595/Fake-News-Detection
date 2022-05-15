from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('output', views.news_url)
]