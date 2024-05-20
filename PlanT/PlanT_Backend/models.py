# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
# import datetime
# from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models


class User(models.Model):
    user_email = models.EmailField(primary_key=True)
    last_login = models.DateTimeField(auto_now=True)


# class City(models.Model):
#     city_id = models.PositiveSmallIntegerField(primary_key=True)
#     city_name = models.CharField(max_length=10)
#     city_tag = models.ForeignKey(Tag, on_delete=models.CASCADE)
    
    
class Tag(models.Model):
    tag_id = models.PositiveSmallIntegerField(primary_key=True)
    tag_name = models.CharField(max_length=20)
    

class Place(models.Model):
    place_id = models.PositiveIntegerField(primary_key=True)
    place_name = models.CharField(max_length=100)
    TYPE_CHOICES = (
        (1, '식사'),
        (2, '카페'),
        (3, '숙박'),
        (4, '관광'),
    )
    place_type = models.PositiveSmallIntegerField(choices=TYPE_CHOICES)
    place_time = models.PositiveSmallIntegerField()
    place_latitude = models.DecimalField(max_digits=8, decimal_places=6)
    place_longitude = models.DecimalField(max_digits=9, decimal_places=6)
    place_detail = models.JSONField(default=None)
    place_tag = models.ForeignKey(Tag, on_delete=models.CASCADE)


class Trip(models.Model):
    trip_id = models.AutoField(primary_key=True)
    trip_start = models.DateField()
    trip_end = models.DateField()
    STATE_CHOICES = (
        (1, '예정'),
        (2, '진행중'),
        (3, '완료'),
    )
    trip_state = models.PositiveSmallIntegerField(choices=STATE_CHOICES)
    trip_user = models.ForeignKey(User, on_delete=models.CASCADE)


class Plan(models.Model):
    plan_id = models.AutoField(primary_key=True)
    plan_date = models.DateField()
    plan_trip = models.ForeignKey(Trip, on_delete=models.CASCADE)


class Route(models.Model):
    route_id = models.AutoField(primary_key=True)
    route_starttime = models.DateTimeField()
    route_endtime = models.DateTimeField()
    # route_detail = models.JSONField(blank=True, null=True, default=None)
    route_start = models.ForeignKey(Place, on_delete=models.CASCADE, related_name='route_start_place')
    route_end = models.ForeignKey(Place, on_delete=models.CASCADE, related_name='route_end_place')
    route_plan = models.ForeignKey(Plan, on_delete=models.CASCADE)
