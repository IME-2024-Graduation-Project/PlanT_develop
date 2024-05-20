from django.urls import path
from .views import GoogleLogin, GoogleLogout, GoogleState, Sender

# from rest_framework_simplejwt.views import (
#     TokenRefreshView,
# )

urlpatterns = [
    # path('token/', views.MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # path('register/', views.RegisterView.as_view(), name='auth_register'),
      path('send/', Sender),
      path('account/login/', GoogleLogin, name='google_login'),
      path('account/logout/', GoogleLogout, name='google_logout'),
      path('account/state/', GoogleState, name='google_state'),
    # path('trip/', views.TripMaker),
    # path('route/', views.RouteMaker),
    # path('clust/', views.ClusterMaker),
    # path('plan/', views.Planner),
]