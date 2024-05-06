from  django.urls import path
from . import views



urlpatterns = [
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('logout', views.logout, name='logout'),
    path('contact', views.contact, name='contact'),
    path('about', views.about, name='about'),
    path('news', views.news, name='news'),
    path('data',views.data,name='data'),
    path('predict',views.predict,name='predict'),
    path('preprocess',views.preprocess,name='preprocess'),
    path('visualize',views.visualize,name='visualize'),
    path('prediction',views.prediction,name='prediction'),
    path('review',views.comment_Review,name='review'),
    path('view_review',views.view_review,name='view_review'),
     

]