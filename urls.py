from django.urls import path
from . import views

urlpatterns = [
	path('', views.survey, name = 'survey'),
    path('genre_preload', views.genre_preload2, name = 'genre_preload'),
    path('object_detection_preload', views.vgg_model_view, name = 'object_detection_preload'),
    path('genre_test_multiple', views.genre_test_multiple, name = 'genre_test_multiple'),
    path('successpage', views.successpage, name = 'successpage'),
    path('object_test_multiple', views.object_test_multiple, name = 'object_test_multiple'),
    path('survey_submission', views.SurveyStepsFormSubmission.as_view(), name= 'survey'),
    path('survey',views.question, name= 'survey'),
    path('survey2',views.question2, name= 'survey2'),
    path('survey3',views.question3, name= 'survey3'),       
    ]