import joblib
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd


def create_evaluation_table(results_dict):
    rows = []
    for model_name, result in results_dict.items():
        rows.append({
            'Model': model_name,
            'MAE': result['mae'],
            'MSE': result['mse'],
            'RMSE': result['rmse'],
            'R²': result['r2']
        })
    return pd.DataFrame(rows).set_index("Model") 
     
    

# interactive_predictor.py
def predict_next_day_from_input(input_dict, scaler_feature, model_paths, target_column='Close'):
    input_df = pd.DataFrame([input_dict])
    scaled_input = scaler_feature.transform(input_df)

    predictions = {}
    inverse_predictions = {}
    for name, path in model_paths.items():
        model = joblib.load(path)
        scaled_pred = model.predict(scaled_input).item()
        predictions[name] = round(scaled_pred, 4)

        # Inverse scale using dummy with predicted value
        dummy = scaled_input.copy()
        target_index = input_df.columns.get_loc(target_column)
        dummy[0][target_index] = scaled_pred
        original = scaler_feature.inverse_transform(dummy)[0][target_index]
        inverse_predictions[name] = round(float(original), 4)

    return {
    
        "original predictions": inverse_predictions
    }

def display_prediction_interface(scaler_feature, model_paths):
    # Input widgets
    open_input = widgets.FloatText(description='Open:')
    high_input = widgets.FloatText(description='High:')
    low_input = widgets.FloatText(description='Low:')
    close_input = widgets.FloatText(description='Close:')
    volume_input = widgets.FloatText(description='Volume:')

    predict_button = widgets.Button(description="Predict Next Day Price")
    output = widgets.Output()

    def on_predict_clicked(b):
        with output:
            output.clear_output()
            input_data = {
                'Open': open_input.value,
                'High': high_input.value,
                'Low': low_input.value,
                'Close': close_input.value,
                'Volume': volume_input.value
            }
            results = predict_next_day_from_input(input_data, scaler_feature, model_paths)
            print(" Predictions for Next Day:")
            for key, value in results.items():
                print(f"$ {key}: {value}")

    predict_button.on_click(on_predict_clicked)

    display(open_input, high_input, low_input, close_input, volume_input, predict_button, output)


def predict_next_close_lstm(test_df, feature_scaler_lstm, target_scaler_lstm, model_path, i=0):
   
    # Load model
    model_lstm = load_model(model_path)

    # Prepare input
    input_seq = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i:i+5].values
    scaled_seq = feature_scaler_lstm.transform(input_seq).reshape(1, 5, 5)

    # Predict
    log_pred = model_lstm.predict(scaled_seq)
    predicted_close = target_scaler_lstm.inverse_transform(log_pred)[0][0]

    # Actual Close
    actual_close = test_df.iloc[i+5]['Close']

    return round(actual_close, 2), round(predicted_close, 2)










