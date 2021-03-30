from django.shortcuts import render
import os.path
# Create your views here.

import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np
import pandas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class_a_path = os.path.join(BASE_DIR,"finalisedtitanic_model.sav")
# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(["POST"])
def predict_servivedtype(request):
    try:
        
        pclass = request.data.get('pclass',None)
        male= request.data.get('male',None)
        age = request.data.get('age',None)
        siblings_Spouses = request.data.get('siblings_Spouses',None)
        parents_Children = request.data.get('parents_Children',None)
        fare = request.data.get('fare',None)
        fields = [male,pclass,age,siblings_Spouses,parents_Children,fare]
        if not None in fields:
            #Datapreprocessing Convert the values to float
            pclass = float(pclass)
            male = bool(male)
            age = float(age)
            siblings_Spouses = float(siblings_Spouses)
            parents_Children = float(parents_Children)
            fare = float(fare)
            result = [pclass,male,age,siblings_Spouses,parents_Children,fare]
            #Passing data to model & loading the model from disks
            model_path = 'C:/Users/Ap-Android/Desktop/mldeployement/model_deploy/model_deploy/titanic_model.pkl'
            classifier = pickle.load(open(model_path, 'rb'))
           
            prediction = classifier.predict([result])
           # conf_score =  np.max(classifier.predict_proba([result]))*100
            predictions = {
                'error' : '0',
                'message' : 'Successfull',
                'prediction' : prediction,
                'confidence_score' : '0'
            }
        else:
            predictions = {
                'error' : '1',
                'message': 'Invalid Parameters'                
            }
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    
    return Response(predictions)
