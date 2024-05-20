from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib.auth import get_user_model
from .models import Place, Plan, Trip, Route, Tag, User
from django.views.decorators.csrf import csrf_exempt
import json
# from django.core.serializers import serialize
# from datetime import date, timedelta, datetime
# from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
# from dj_rest_auth.registration.views import SocialLoginView
# from google.auth.transport import requests
# from google.oauth2 import id_token
# from django.conf import settings
# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import login, logout


@api_view(['POST'])
def GoogleLogin(request):
    if request.method == 'POST':
        email = request.data.get('email', None)
        if email:
            # 수정된 부분: User 모델 사용
            user, created = User.objects.get_or_create(user_email=email)
            login(request, user)
            return Response({'id': user.user_email}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Email address is required.'}, status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response({'error': 'Method not allowed.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def GoogleLogout(request):
    if request.method == 'POST':
        # 로그아웃 요청 처리
        logout(request)
        return Response({'message': 'Logout successful'}, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'Method not allowed.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def GoogleState(request):
    if request.method == 'POST':
        if request.session:
            user_email = request.session.get('user_email')
        else:
            user_email = ""
        return Response(user_email, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'Method not allowed.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# @csrf_exempt
# def sender(request):
#     city = City.objects.all().values_list("city_name", flat=True)
# #    plc1 = Place.objects.all().values_list("city_id", flat=True)
# #    plc2 = Place.objects.all().values_list("city_id", flat=True)
# #    Plan = City.objects.all().values_list("city_id", flat=True)
    
#     return JsonResponse(list(city), safe=False)


# @csrf_exempt
# def reciever(request):
#     if request.method == 'POST':
#         data = json.loads(request.body.decode('utf-8'))  # JSON 데이터를 파이썬 객체로 변환
#         option_value = data.get('ops')
#         if option_value:
#             option = str(option_value)
#             cid = City.objects.get(city_name=option)
#             option = Place.objects.filter(city=cid).values_list('place_id','place_name')
#             dict = {}
#             for i in option:
#                 dict[i[0]] = i[1]
#             #serialized_option = serialize('json', option, fields=('place_name',))
#             return JsonResponse(dict, safe=False)
#         else:
#             return JsonResponse({'error': 'Option value is required'}, status=400)
        
#     else:
#         return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)



# class MyTokenObtainPairView(TokenObtainPairView):
#     serializer_class = MyTokenObtainPairSerializer


# class RegisterView(generics.CreateAPIView):
#     queryset = User.objects.all()
#     permission_classes = (AllowAny,)
#     serializer_class = RegisterSerializer


# @api_view(['GET'])
# def getRoutes(request):
#     routes = [
#         '/api/token/',
#         '/api/register/',
#         '/api/token/refresh/'
#     ]
#     return Response(routes)


#class recommender:
#    def __init__(self, ):


def Filter(table, filter_dict, use_fields):
    
    if not filter_dict:
        filtered_objects = table.objects.values(*use_fields)
    
    filtered_objects = table.objects.filter(**filter_dict).values(*use_fields)   
    return filtered_objects


@csrf_exempt        
def Sender(request):
    
    if request.method == 'POST':
        
        data = json.loads(request.body.decode('utf-8'))  # JSON 데이터를 파이썬 객체로 변환
        table = data.get('table')
        fields = data.get('fileds')
        values = 0
        filter_dict = 0
            
        if fields != 'all':
            values = data.get('values')
            filter_dict = dict(zip(fields, values))
        
        
        if table and fields:
            
            if table == 'tag' and fields == 'all': # 전체 태그
                send = Tag.objects.values('tag_id', 'tag_name')
                
            # elif table== 'city': # 도시 필터링
            #     use_fields = ['city_id', 'city_name'] 
            #     send = Filter(City, filter_dict, use_fields)

            elif table == 'place': # 장소 필터링
                use_fields = ['place_id', 'place_name']
                send = Filter(Place, filter_dict, use_fields)
                
                if data.get("selected_plc"): # 선택된 장소 제외 필터링
                    selected_plc = data.get("selected_plc")
                    poss_plc = Place.objects.exclude(id__in = selected_plc)
                    send = Filter(poss_plc, filter_dict, use_fields)
                    
            elif table == 'route': # 루트 필터링
                use_fields = ['route_id','route_time', 'route_co2']
                send = Filter(Route, filter_dict, use_fields) # route_plan = plan_id
            
            elif table == 'detail_route': # 상세 루트 정보
                route = data.get('route')
                send = {
                    'rount_startplc' : Route.objects.get(pk=route).route_start,
                    'rount_endplc' : Route.objects.get(pk=route).route_end,
                    'rount_starttime' : Route.objects.get(pk=route).route_start,
                    'rount_endtime' : Route.objects.get(pk=route).route_end,
                }
                # send['rount_startplc'] = Route.objects.get(pk=route).route_start  
                # send['rount_endplc'] = Route.objects.get(pk=route).route_end
                # send['rount_starttime'] = Route.objects.get(pk=route).route_start  
                # send['rount_endtime'] = Route.objects.get(pk=route).route_end
                
                
            return JsonResponse(send, safe=False)
        
        else:
            return JsonResponse({'error': 'value is required'}, status=400)

    
    else:
        return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)       

    
# def TripMaker(request):    
    
#     if request.method == 'POST':
        
#         data = json.loads(request.body.decode('utf-8'))  # JSON 데이터를 파이썬 객체로 변환
#         start_date = data.get('start_date')
#         end_date = data.get('end_date')
#         move_date = data.get('move_date')
#         from_city = data.get('from_city')
#         to_city = data.get('to_city')
        
#         new_trip = Trip(trip_start = start_date, 
#                         trip_end = end_date,
#                         trip_state = 1,
#                         trip_ecolevel = data.get('poss_time'),
#                         trip_posstime = data.get('eco_lev'),
#                         trip_user = 'user1')
        
#         new_trip.save()
        
#         current_date = start_date
        
#         plan_count = (end_date - start_date).days 
#         plc_count = plan_count * 2 # 고를 수 있는 장소 수
    
#         stay_city = from_city
#         plan_list = []
#         move_city_list = [stay_city]
        
#         for i in range(plan_count + 1):
               
#             if current_date > move_date:
#                 stay_city = to_city
#                 move_city = to_city
#                 move_plan = 1
#                 if new_plan:
#                     move_plan = new_plan.plan_id
#                 plc_count -= 2
#                 move_plan_index = i+1          

#             new_plan = Plan(plan_date = current_date,
#                             plan_trip = new_trip.trip_id,
#                             plan_city = stay_city)
            
#             new_plan.save()
#             plan_list.append(new_plan.plan_id)
#             current_date += timedelta(days=1)
            
#             move_city_list.append(move_city)
        
#         return JsonResponse({'message': 'New trip is created with plans',
#                              'move_plan': move_plan,
#                              'move_plan_index' : move_plan_index,
#                              'move_city_list' : move_city_list,
#                              'plan_list' : plan_list,
#                              'plc_count' : plc_count},
#                             safe=False)
        
#     else:
#         return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)     
 

# def RouteMaker(request):
    
#     if request.method == 'POST':
        
#         data = json.loads(request.body.decode('utf-8'))
#         start_plc = data.get('start_plc')
#         end_plc = data.get('end_plc')
#         move_plan = data.get('move_plan')
        
#         origin_date = Plan.objects.get(pk=move_plan).plan_date

#         new_route = Route(route_type = 3,
#                           route_transport = 3,
#                           route_starttime = origin_date.replace(hour=8, minute=0, second=0, microsecond=0),
#                           route_endtime = origin_date.replace(hour=23, minute=0, second=0, microsecond=0),
#                           route_time = 15,
#                           route_co2 = 0,
#                           route_detail = None,
#                           route_start = start_plc,
#                           route_end = end_plc,
#                           route_plan = move_plan)
        
#         new_route.save()
        
#         return JsonResponse({'message': 'New move plan is created'}, safe=False)
        
#     else:
#         return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)  


# def ClusterMaker(request):
     
#     if request.method == 'POST':
        
#         data = json.loads(request.body.decode('utf-8'))
#         plc_list = data.get('plc_list')
#         cluster_count = data.get('clst_count')
        
#         plc_dict = {}
        
#         return JsonResponse(plc_dict, safe=False)
        
#     else:
#         return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)
    
    

# def PlanClassifier(request):
    
#     if request.method == 'POST':
        
#         data = json.loads(request.body.decode('utf-8'))
#         plan_dict = data.get('plan_dict')
        
    
#         tag_list = Tag.objects.values_list('tag_id', flat=True)
#         tag_dict = {}
        
#         for tag in tag_list:
#             tag_dict[tag] = 0
                
#         for i in range(1,len(plan_dict)+1):
            
#             for j in plan_dict[i]:
                
#                 if j in tag_list:
#                     tag_dict[j] += 1
                    
                
        
#         return plan_dict, plc, pln 
    
#     else:
#         return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)   
          
            
# def AssignTag(eco_lev, tag_dict):

#     if eco_lev == 0:
#         tag_per = 0
        
#     elif eco_lev == 1:
#         tag_per = 20
        
#     elif eco_lev == 2:
#         tag_per = 40
        
#     elif eco_lev == 3:
#         tag_per = 60
        
#     elif eco_lev == 4:
#         tag_per = 80
        
#     else:
#         tag_per = 100
        
#     for i in tag_dict:
#         tag_dict[i] /= tag_per
        
#     return tag_dict


# def Coordinate(plc_id):
    
#     plc = Place.objects.get(pk=plc_id)
        
#     return (plc.place_latitude, plc.place_longitude)
             
            
# def Planner(request):
    
#     if request.method == 'POST':
        
#         data = json.loads(request.body.decode('utf-8'))
#         plan_dict = data.get('plan_dict')
#         tag_dict = data.get('tag_dict')
#         eco_lev = Plan.objects.get(pk=plan_dict[1][0]).plan_trip.trip_ecolevel
#         tag_dict = AssignTag(eco_lev, tag_dict)
    
#         i, j = 1, 0
        
#         while i <= len(plan_dict) and j < len(plan_dict[i][1]):
            
#             convex_hull = [Coordinate(k) for k in plan_dict[i][1] if k >= 10]
                    
#             if plan_dict[i][1][j] - 10 < 0:
#                 city = Plan.objects.get(pk=plan_dict[i][0]).plan_city.city_id
#                 place_set = Place.objects.filter(place_city=city, place_tag=plan_dict[i][1][j]).values_list('place_id', flat=True)
                
#                 input = {1 : plan_dict[i][1][j-1], # 출발지
#                          2 : convex_hull, # 폴리곤 만드는데 사용되어야 할 좌표들(위도, 경도), 해당 일정에서 특정된 장소들, 만약 1개라면 num1_in_out 사용하도록
#                          3 : place_set} # 해당 지역, 해당 태그로 1차 필터링된 장소들(in_out의 대상)
                
#                 # input 보내서 접근 가능한 장소(id) 리스트로 받아오기
#                 # plc_poss = in_out(input)
#                 plc_poss = Place.objects.filter(place_tag = plan_dict[i][j]) # 나중에 받으면 교체
                        
#                 # 친환경 레벨에 해당하는 비율만큼 할당된 경우 일반 장소로만 추천, 남아있으면 친환경 장소로만 추천
#                 plc_poss_eco = [plc for plc in plc_poss if Place.objects.get(pk=plc).place_eco]
#                 plc_poss_noneco = [plc for plc in plc_poss if not Place.objects.get(pk=plc).place_eco]
#                 eco_state = 0
                
#                 if tag_dict[plan_dict[i][1][j]] > 0 and plc_poss_eco:
                    
#                     tag_dict[plan_dict[i][1][j]] -= 1
#                     eco_state = 1
                    
#                 coordinated_plc_poss = [Coordinate(plc) for plc in plc_poss_noneco]
                
#                 input = {1 : Coordinate(plan_dict[i][1][j-1]), # 출발지 좌표(x,y)
#                          2 : coordinated_plc_poss, # 목적지 후보들 좌표(x,y) 리스트
#                          3 : plc_poss_noneco} # 목적지 후보들 id(위 좌표 리스트와 순서쌍)
                
#                 if eco_state:
#                     coordinated_plc_poss_eco = [Coordinate(plc) for plc in plc_poss_eco]
                    
#                     input_eco = {1 : Coordinate(plan_dict[i][1][j-1]), # 출발지 좌표(x,y)
#                                 2 : coordinated_plc_poss_eco, # 목적지 후보들 좌표(x,y) 리스트
#                                 3 : plc_poss_eco} # 목적지 후보들 id(위 좌표 리스트와 순서쌍)
                    
#                     eco_route_info = 1
                    
#                     last_route_id = Route.objects.values_list('route_id',flat=True).order_by('-route_id')[0]
#                     new_route_id_eco = last_route_id + 1
                    
#                     last_plan_route = Route.objects.filter(route_plan=plan_dict[i][0]).values_list('route_id',flat=True)
                    
#                     if last_plan_route:
#                         last_plan_route_id = last_plan_route.order_by('-route_id')[0]
#                         start_time = Route.objects.get(pk=last_plan_route_id).route_endtime
                    
#                     else:
#                         start_time = Plan.objects.get(pk=plan_dict[i][0]).plan_date + datetime.timedelta(hours=8, minutes=0, seconds=0, microseconds=0)
                    
#                     end_time = start_time + timedelta(minutes=eco_route_info[2])
                
#                 # 루트 플랜에 저장하기
#                     new_route_eco = Route(route_type = 1,
#                                           route_transport = eco_route_info[1],
#                                           route_starttime = start_time,
#                                           route_endtime = end_time,
#                                           route_time = eco_route_info[2],
#                                           route_co2 = eco_route_info[3],
#                                           route_detail = None,
#                                           route_start = plan_dict[i][1][j],
#                                           route_end = eco_route_info[0],
#                                           route_plan = plan_dict[i][0])
                
#                     new_route_eco.save()
                    
#                 # 보내서 최단거리 장소(id),이동수단,소요시간,탄소배출량 받기
#                 route_info = 1
#                 plc_id = 11
#                 plc_move = 3
#                 plc_time = 30
#                 plc_co2 = 30 # 정해지면 지우기
                
#                 # 루트 플랜에 저장하기
#                 last_plan_route = Route.objects.filter(route_plan=plan_dict[i][0]).values_list('route_id',flat=True)
                    
#                 if last_plan_route:
#                     last_plan_route_id = last_plan_route.order_by('-route_id')[0]
#                     start_time = Route.objects.get(pk=last_plan_route_id).route_endtime
                    
#                 else:
#                     start_time = Plan.objects.get(pk=plan_dict[i][0]).plan_date + datetime.timedelta(hours=8, minutes=0, seconds=0, microseconds=0)
                    
#                 end_time = start_time + timedelta(minutes=route_info[2])
                
#                 # 루트 플랜에 저장하기
#                 new_route = Route(route_type = 2,
#                                   route_transport = route_info[1],
#                                   route_starttime = start_time,
#                                   route_endtime = end_time,
#                                   route_time = route_info[2],
#                                   route_co2 = route_info[3],
#                                   route_detail = None,
#                                   route_start = plan_dict[i][1][j],
#                                   route_end = route_info[0],
#                                   route_plan = plan_dict[i][0])
            
#                 new_route.save()
                
#             else: # 장소 -> 장소(지정경로)  
                
#                 confirmed_route_info = APIRoute(Coordinate(plan_dict[i][1][j-1]), Coordinate(plan_dict[i][1][j]))  
                
#                 last_route_id = Route.objects.values_list('route_id',flat=True).order_by('-route_id')[0]
#                 new_confirmed_route_id = last_route_id + 1
                    
#                 last_plan_route = Route.objects.filter(route_plan=plan_dict[i][0]).values_list('route_id',flat=True)
                    
#                 if last_plan_route:
#                     last_plan_route_id = last_plan_route.order_by('-route_id')[0]
#                     start_time = Route.objects.get(pk=last_plan_route_id).route_endtime
                    
#                 else:
#                     start_time = Plan.objects.get(pk=plan_dict[i][0]).plan_date + datetime.timedelta(hours=8, minutes=0, seconds=0, microseconds=0)
                    
#                 end_time = start_time + timedelta(minutes=confirmed_route_info[2])
                
#                 # 루트 플랜에 저장하기
#                 new_confirmed_route = Route(route_id = new_confirmed_route_id,
#                                             route_type = 3,
#                                             route_transport = confirmed_route_info[1],
#                                             route_starttime = start_time,
#                                             route_endtime = end_time,
#                                             route_time = confirmed_route_info[2],
#                                             route_co2 = confirmed_route_info[3],
#                                             route_detail = None,
#                                             route_start = plan_dict[i][1][j-1],
#                                             route_end = plan_dict[i][1][j],
#                                             route_plan = plan_dict[i][0])
            
#                 new_confirmed_route.save()    
                            
#             j += 1 
                    
#             if j == len(plan_dict[i][1]): # 해당 일자의 장소가 모두 정해지면 다음 장소로 변경

#                 j = 0
#                 i += 1
        
#         trip = Plan.objects.get(pk=plan_dict[0][0]).plan_trip.trip_id
#         plans_list = Plan.objects.filter(plan_trip=trip).values_list('plan_id', flat=True).order_by('plan_id')
                
#         return JsonResponse(plans_list, safe=False)
        
#     else:
#         return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)        
                
                    
                    

            




