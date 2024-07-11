from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('best_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    
    # Preprocess the input data
    df_scaled = scaler.transform(df)
    
    # Make predictions
    predictions = model.predict(df_scaled)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
