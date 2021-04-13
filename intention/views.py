from django.shortcuts import render
import json
import numpy as np
import tensorflow as tf
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
@csrf_exempt
def predict(request):

    #If Post method
    if request.POST is not None:

        var = json.loads(request.body)

        Administrative = float(var['Administrative'])
        Administrative_Duration = float(var['Administrative_Duration'])
        Informational = float(var['Informational'])
        Informational_Duration = float(var['Informational_Duration'])
        ProductRelated = float(var['ProductRelated'])
        ProductRelated_Duration = float(var['ProductRelated_Duration'])
        BounceRates = float(var['BounceRates'])
        ExitRates = float(var['ExitRates'])
        PageValues = float(var['PageValues'])
        SpecialDay = float(var['SpecialDay'])
        OperatingSystems = float(var['OperatingSystems'])
        Browser = float(var['Browser'])
        Region = float(var['Region'])
        TrafficType = float(var['TrafficType'])
        Dec = float(var['Dec'])
        Feb = float(var['Feb'])
        Jul = float(var['Jul'])
        June = float(var['June'])
        Mar = float(var['Mar'])
        May = float(var['May'])
        Nov = float(var['Nov'])
        Oct = float(var['Oct'])
        Sep = float(var['Sep'])
        Other = float(var['Other'])
        Returning_Visitor = float(var['Returning_Visitor'])
        Weekend = float(var['Weekend'])

        #Loading Scaler and Model
        scaler = joblib.load('scalar.sav')
        model = tf.saved_model.load("content")

        print(list(model.signatures.keys()))
        infer = model.signatures["serving_default"]
        print(infer.structured_outputs)

        data = scaler.transform(np.array([[Administrative,Administrative_Duration,Informational,
                                              Informational_Duration,ProductRelated,ProductRelated_Duration,
                                              BounceRates,ExitRates,PageValues,SpecialDay,OperatingSystems,
                                              Browser,Region,TrafficType,Dec,Feb,Jul,June,Mar,May,Nov,Oct,
                                              Sep,Other,Returning_Visitor,Weekend]], np.float32))

        print("My data Type is:",type(data))

        result = infer(tf.constant(data))
        print("My Prediction is:",result['dense_2'][0][0])

        if result['dense_2'][0][0] <= .5:
            output = 'Less Likely to Buy the Product'
        else:
            output = 'More Like to Buy the product!'

        return JsonResponse({"pred": output}, safe=False)



