from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("audit/", views.run_audit, name="run_audit"),
]
