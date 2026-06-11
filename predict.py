import joblib
import numpy as np

def predict():

    model = joblib.load("models/best_model.pkl")

    sample = np.array([
        [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
    ])

    result = model.predict(sample)

    if result[0] == 1:
        print("Heart Disease Predicted")
    else:
        print("No Heart Disease Predicted")