from joblib import load
import os
import pandas as pd

MODEL_DIR = 'model_risk.joblib' 
COLUMNS = ['age', 'years_on_the_job', 'nb_previous_loans', 'avg_amount_loans_previous', 'flag_own_car']

def model_fn(model_dir):
    model = load(os.path.join(model_dir, MODEL_DIR))
    return model

def input_fn(request_body):
    request_array = eval(request_body)
    request_df = pd.DataFrame(request_array, columns=COLUMNS)
    return request_df

def predict_fn(input_data, model):
    
    prediction = model.predict(input_data)
    return prediction
