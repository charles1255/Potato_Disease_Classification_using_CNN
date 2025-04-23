from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model('model.keras')

# Define class names for prediction labels
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Model parameters (should match the training setup)
BATCH_SIZE = 32
IMAGE_SIZE = 255  # Ensure the model was trained with this size
CHANNEL = 3
EPOCHS = 20


# Function to preprocess the image and make predictions
def predict(img):
    # Convert image to array and expand dimensions to match model input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Model expects batch dimension

    # Get model predictions
    predictions = model.predict(img_array)

    # Determine the class with the highest probability
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)  # Convert to percentage

    return predicted_class, confidence


# Home route for image upload and classification
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']  # Get the uploaded file

        # Validate if the file is selected
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Validate if the file format is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Secure the filename
            filepath = os.path.join('static', filename)  # Save path
            file.save(filepath)  # Save the file in static folder

            # Load the image with the correct target size (255x255)
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Perform prediction
            predicted_class, confidence = predict(img)

            # Render the template with prediction results
            return render_template('index.html', image_path=filepath, actual_label=predicted_class,
                                   predicted_label=predicted_class, confidence=confidence)

    return render_template('index.html', message='Upload an image')


# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
