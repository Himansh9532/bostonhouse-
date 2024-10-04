from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler
regmodel = pickle.load(open("linear_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # Load the scaler used during training

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    # Get JSON data from the request
    data = request.json
    print(data)  # Print the received data for debugging

    # Extract features from the nested data structure
    features = data['data']  # Get the 'data' key

    # Convert the features dictionary to a NumPy array
    new_data = np.array(list(features.values())).reshape(1, -1)  # Correctly reshaping the input

    # Make a prediction using the loaded model
    output = regmodel.predict(new_data)

    # Return the prediction as JSON
    response = {
        "prediction": output[0]  # Extract the prediction value
    }
    
    return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the form data as a dictionary and convert values to float
    data = request.form.to_dict()  # Get the form data as a dictionary
    data = [float(value) for value in data.values()]  # Convert values to float

    # Standardize the input data using the loaded scaler
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)  # Print the standardized input for debugging

    # Make the prediction
    output = regmodel.predict(final_input)[0]
     # Convert the prediction to INR (update the exchange rate as needed)
    exchange_rate = 83.0  # Example exchange rate (1 USD = 83 INR)
    price_in_inr = output * exchange_rate

    return render_template("home.html",  prediction_text="The House Price Prediction is ₹{:.2f}".format(output).format(output))

if __name__ == "__main__":
    app.run(debug=True)
