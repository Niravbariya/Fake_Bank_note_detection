from flask import Flask, render_template,request
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("models/model.h5")
scaler = pickle.load(open("models/scaler.pkl", 'rb'))

# Prediction function
def make_prediction(input_data):
    # Preprocess input data (apply scaling)
    input_data_scaled = scaler.transform(input_data)

    # Use the trained model to predict the class
    predictions = model.predict(input_data_scaled)

    # Convert prediction to binary (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)

    return predicted_classes

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['GET', 'POST'])  # ✅ Fixed: methods
def predict():
    if request.method == 'POST':
        # Get form inputs
        # VWTI = float(request.form['VWTI'])
        # SWTI = float(request.form['SWTI'])
        # CWTI = float(request.form['CWTI'])
        # EI = float(request.form['EI'])
        
        VWTI = float(request.form['VWTI'])
        SWTI = float(request.form['SWTI'])
        CWTI = float(request.form['CWTI'])
        EI = float(request.form['EI'])


        # Prepare input for prediction
        input_data = np.array([[VWTI, SWTI, CWTI, EI]])

        # Make prediction
        result = make_prediction(input_data)

        # Interpret result
        output = 'real' if result[0][0] == 1 else 'fake'

        return render_template('index.html', prediction=output)

    return render_template('index.html', prediction=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
