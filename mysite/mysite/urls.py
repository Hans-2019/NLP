"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import function.app01 as app01
import function.app02 as app02
import function.app03 as app03

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('app01/home', app01.home),
    path('app01/intro',app01.intro),
    path('app01/define',app01.define),
    path('app01/loss_func',app01.lossf),
    path('app02/apply',app02.apply),
    path('app02/home',app02.home),
    path('app02/graph',app02.graph)
]
