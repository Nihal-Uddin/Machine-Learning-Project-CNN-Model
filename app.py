from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model('animal_disease_model.h5')

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


labels = [
    'cow_FMD', 'cow_Healthy', 'cow_Lumpy_Skin', 'cow_Mastitis',
    'dog_Demodicosis', 'dog_Dermatities', 'dog_Fungal_infections',
    'dog_Healthy', 'dog_Hypersensitivity', 'dog_Ringworm',
    'goat_Healthy_goat', 'goat_Unhealthy_goat'
]


label_to_disease = {
    'cow_FMD': 'FMD',
    'cow_Healthy': 'Healthy',
    'cow_Lumpy_Skin': 'Lumpy Skin',
    'cow_Mastitis': 'Mastitis',
    'dog_Demodicosis': 'Demodicosis',
    'dog_Dermatities': 'Dermatitis',
    'dog_Fungal_infections': 'Fungal Infection',
    'dog_Healthy': 'Healthy',
    'dog_Hypersensitivity': 'Hypersensitivity',
    'dog_Ringworm': 'Ringworm',
    'goat_Healthy_goat': 'Healthy',
    'goat_Unhealthy_goat': 'Unhealthy'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = load_img(filepath, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    # Get readable disease name
    predicted_label = labels[class_index]
    disease_name = label_to_disease.get(predicted_label, predicted_label)

    return render_template('result.html', prediction=disease_name, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
