from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_energy_consumption_api, name='energy_consumption_api')
]
