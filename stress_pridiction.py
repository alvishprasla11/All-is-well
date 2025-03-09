import joblib

def stressLevel(input):

    # Load and use the model later
    loaded_model = joblib.load("stress_prediction_model.pkl")
    new_prediction = loaded_model.predict(input)

    return new_prediction

input =[[30, 1, 150, 2, 7.0, 4.0, 0, 2.0, 4.0, 0, 0, 8, 1.0, 5, 1, 0, 120, 180, 90]]
new_prediction= stressLevel(input)
print("Predicted Stress Level:", new_prediction)