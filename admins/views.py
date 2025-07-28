from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import SneakerPricePredictor
import json
import os
from django.conf import settings

predictor = SneakerPricePredictor()

def train_model_view(request):
    if request.method == 'POST':
        data_path = os.path.join(settings.MEDIA_ROOT, 'StockX-Data-Contest-2019-3.csv')
        success, result = predictor.train_model(data_path)
        
        if success:
            messages.success(request, f'Model trained successfully with MAE: ${result:,.2f}')
        else:
            messages.error(request, f'Training failed: {result}')
        
        return redirect('ml_page')
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def predict_price_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            required_fields = [
                'Brand', 'Sneaker Name', 'Retail Price', 
                'Order Date', 'Release Date', 'Buyer Region'
            ]
            
            if not all(field in data for field in required_fields):
                return JsonResponse({'error': 'Missing required fields'}, status=400)
            
            prediction = predictor.predict_price(data)
            return JsonResponse({'predicted_price': prediction})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

def ml_page(request):
    return render(request, 'users/ml.html')
