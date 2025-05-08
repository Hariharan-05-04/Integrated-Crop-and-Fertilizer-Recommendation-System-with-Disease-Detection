import os
import pickle

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models and encoders for crop recommendation
with open("rf_crop_recommendation.pkl", "rb") as model_file:
    crop_model = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as encoder_file:
    crop_label_encoders = pickle.load(encoder_file)

# Load models and encoders for fertilizer recommendation
fertilizer_model = joblib.load("fertilizer_rf_model.pkl")
fertilizer_label_encoders = joblib.load("ferti_encoder.pkl")

# Load plant disease model
plant_disease_model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Define dropdown options for crop prediction
states = crop_label_encoders["State"].classes_.tolist()
districts = crop_label_encoders["District"].classes_.tolist()

# Define dropdown options for fertilizer prediction
soil_types = fertilizer_label_encoders['Soil_Type'].classes_.tolist()
crop_types = fertilizer_label_encoders['Crop_Type'].classes_.tolist()

# Disease class names
class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
               'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew',
               'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy',
               'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
               'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
               'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
               'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
               'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
               'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
               'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
               'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
               'Tomato___healthy']


def predict_fertilizer(input_data):
    feature_order = ["Temparature", "Humidity", "Moisture", "Soil_Type", "Crop_Type",
                     "Nitrogen", "Potassium", "Phosphorous"]
    df = pd.DataFrame([input_data])[feature_order]

    for col in ['Soil_Type', 'Crop_Type']:
        df[col] = fertilizer_label_encoders[col].transform(df[col])

    prediction = fertilizer_model.predict(df)
    predicted_fertilizer = fertilizer_label_encoders['Fertilizer'].inverse_transform(prediction)

    return predicted_fertilizer[0]


def model_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = plant_disease_model.predict(input_arr)
    return np.argmax(predictions)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_crop', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            # Get form data
            state = request.form['State']
            district = request.form['District']
            nitrogen = float(request.form['N'])
            phosphorus = float(request.form['P'])
            potassium = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Encode categorical values
            state_encoded = crop_label_encoders['State'].transform([state])[0]
            district_encoded = crop_label_encoders['District'].transform([district])[0]

            # Prepare feature array
            features = np.array([[nitrogen, phosphorus, potassium, temperature,
                                  humidity, ph, rainfall, state_encoded, district_encoded]])

            # Make prediction
            prediction = crop_model.predict(features)[0]
            crop = crop_label_encoders['label'].inverse_transform([prediction])[0]

            return render_template('predict.html', prediction=crop, states=states,
                                   districts=districts)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('predict.html', states=states, districts=districts)


@app.route('/predict_fertilizer', methods=['GET', 'POST'])
def predict_fertilizer_route():
    if request.method == 'POST':
        input_data = {
            "Temparature": float(request.form['Temparature']),
            "Humidity": float(request.form['Humidity']),
            "Moisture": float(request.form['Moisture']),
            "Soil_Type": request.form['Soil_Type'],
            "Crop_Type": request.form['Crop_Type'],
            "Nitrogen": int(request.form['Nitrogen']),
            "Potassium": int(request.form['Potassium']),
            "Phosphorous": int(request.form['Phosphorous'])
        }
        prediction = predict_fertilizer(input_data)
        return render_template('predict2.html', prediction=prediction,
                               soil_types=soil_types, crop_types=crop_types)

    return render_template('predict2.html', soil_types=soil_types, crop_types=crop_types)


@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict3.html', error='No file uploaded')

        file = request.files['image']
        if file.filename == '':
            return render_template('predict3.html', error='No file selected')

        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Make prediction
            try:
                result_index = model_prediction(filepath)
                prediction = class_names[result_index]

                # Pass both the prediction and the image path to the template
                return render_template('predict3.html',
                                       prediction=prediction,
                                       image_path=os.path.join('uploads', file.filename))
            except Exception as e:
                return render_template('predict3.html', error=str(e))

    return render_template('predict3.html')


if __name__ == '__main__':
    app.run(debug=True)