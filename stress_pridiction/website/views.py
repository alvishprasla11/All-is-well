from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render
import joblib
import numpy as np

# Create your views here.
# View to render the HTML page
def stress_predictor_page(request):
    return render(request, "stress.html")

def stressLevel(input):
    # Load and use the model later
    loaded_model = joblib.load("Model/stress_prediction_model.pkl")
    new_prediction = loaded_model.predict(input)
    return new_prediction

@csrf_exempt  # Temporarily disable CSRF protection (use with caution)
def stress_level_prediction(request):
    if request.method == 'POST':
        # Extract the form data from the POST request
        age = int(request.POST.get('age'))
        gender = int(request.POST.get('gender'))
        occupation = int(request.POST.get('occupation'))
        marital_status = int(request.POST.get('marital_status'))
        sleep_duration = float(request.POST.get('sleep_duration'))
        sleep_quality = int(request.POST.get('sleep_quality'))
        physical_activity = float(request.POST.get('physical_activity'))
        screen_time = float(request.POST.get('screen_time'))
        caffeine_intake = int(request.POST.get('caffeine_intake'))
        alcohol_intake = int(request.POST.get('alcohol_intake'))
        smoking_habit = int(request.POST.get('smoking_habit'))
        work_hours = int(request.POST.get('work_hours'))
        travel_time = int(request.POST.get('travel_time'))
        social_interactions = int(request.POST.get('social_interactions'))
        meditation_practice = int(request.POST.get('meditation_practice'))
        exercise_type = int(request.POST.get('exercise_type'))
        blood_pressure = int(request.POST.get('blood_pressure'))
        cholesterol_level = int(request.POST.get('cholesterol_level'))
        blood_sugar_level = int(request.POST.get('blood_sugar_level'))

        # Prepare the input data for the model
        input_data = [[age, gender, occupation, marital_status, sleep_duration, sleep_quality,
                       physical_activity, screen_time, caffeine_intake, alcohol_intake, smoking_habit,
                       work_hours, travel_time, social_interactions, meditation_practice, exercise_type,
                       blood_pressure, cholesterol_level, blood_sugar_level]]

        print(input_data)
        # Call the stressLevel function and get the prediction
        prediction = stressLevel(input_data)
        
        # Convert the prediction to a native Python int
        prediction_int = int(prediction[0])
        
        # Return the prediction as a response
        return JsonResponse({'stress_level': prediction_int})

    return JsonResponse({'error': 'Invalid request method'}, status=400)