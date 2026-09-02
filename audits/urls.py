from django.urls import path
from . import views

urlpatterns = [
    path(
        "",
        views.dashboard,
        name="dashboard",
    ),
    path(
        "audit/insurance/",
        views.insurance_audit,
        name="insurance_audit",
    ),
    path(
        "audit/public-sector/",
        views.public_sector_audit,
        name="public_sector_audit",
    ),
    path(
        "audit/medicine/",
        views.medical_audit,
        name="medical_audit",
    ),
]
