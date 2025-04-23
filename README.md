
📝 README: Potato Disease Classification Using CNN
------------------------------------------------------
------------------------------------------------------

🌱 Project Overview
This project leverages Convolutional Neural Networks (CNNs) to classify potato leaf diseases (Early Blight, Late Blight, Healthy)
from images. It aims to automate disease detection for farmers, enabling early intervention and reducing crop losses.


📌 Key Features
✅ High Accuracy: Achieves >95% accuracy on test data.
✅ Automated Detection: Eliminates manual inspection errors.
✅ Scalable: Adaptable for large farms via mobile/drone integration.
✅ Real-Time: Processes images with confidence scores.

📂 Project Structure
├── app.py                  # Flask web application  
├── model.keras             # Trained CNN model (Git LFS)  
├── static/                 # Sample images for testing  
│   ├── potato_icon.png  
│   └── disease_samples.JPG  
├── templates/              # HTML frontend  
│   └── index.html  
├── Notebook.ipynb          # Jupyter notebook for model development  
└── requirements.txt        # Python dependencies  

🛠️ Methodology
1. Data Loading & Preprocessing
Dataset: PlantVillage (3 classes: Early Blight, Late Blight, Healthy).

Image Size: 255x255px, RGB channels.

Augmentation: Resizing, rescaling, shuffling.

2. CNN Architecture
  model = Sequential
([
    Conv2D(32, (3,3), activation='relu', input_shape=(255,255,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes
   ])

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

3. Training & Results
Epochs: 20

Batch Size: 32

Test Accuracy: 98.99%

Misclassification: Late Blight → Healthy (88.88% confidence).

🚀 How to Run

Clone the Repository:

git clone https://github.com/charles1255/Potato_Disease_Classification_using_CNN.git
cd Potato_Disease_Classification_using_CNN

Install Dependencies:

pip install -r requirements.txt  # Requires TensorFlow, Flask, etc.

Run the Flask App:

python app.py

Access the web interface at http://localhost:5000.

Test with Custom Images:

Place images in static/ and modify app.py for predictions.


📊 Performance Metrics

Metric	     |      Value
--------------------------
Training     |
Accuracy     |     	97.32%
             |
Validation   |
Accuracy     |    	98.44%
             | 
Test         |
Accuracy     |     	98.99%
             |
Loss	       |       0.0799

🌟 Advantages:
Early disease detection → Prevents yield loss.

Cost-effective compared to lab testing.

Scalable for IoT/edge devices.


⚠️ Limitations:
Misclassification risk for similar-looking diseases.

Requires diverse training data for robustness.


🔮 Future Scope
Mobile app integration for field use.

Expand to other crops (tomatoes, wheat).

Use Vision Transformers (ViTs) for better accuracy.

** 📜License**: [MIT](https://opensource.org/licenses/MIT)

🙏 Acknowledgments
Dataset: [kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

🛠️ Contributions welcome! Submit PRs (Pull Requests) for model improvements or UI enhancements.
📧 Contact: [SHIVAM KUMAR](cgrshivam@gmail.com)











