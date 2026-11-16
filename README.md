# 🤟 Sign Language Detection System

A real-time Sign Language Detection System built using **Python, Convolutional Neural Networks (CNN), TensorFlow, OpenCV, MediaPipe, and Flask**.

The project uses computer vision and deep learning to recognize American Sign Language (ASL) alphabet hand gestures through a webcam and display the predicted letter in real time.

## 📌 Project Overview

Communication is essential for everyone. This project aims to make communication more accessible by developing a computer vision-based system that recognizes hand gestures representing American Sign Language alphabet letters.

The system captures live video from a webcam, detects hand landmarks using MediaPipe, extracts the hand region, and processes the image using a trained CNN model to predict the corresponding sign.

## ✨ Features

- Real-time hand gesture detection using a webcam.
- Hand landmark detection using MediaPipe.
- Image preprocessing using OpenCV.
- Deep learning-based gesture classification using CNN.
- 24-class ASL alphabet classification model.
- Real-time prediction display through a Flask web application.
- Interactive web interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and prediction |
| OpenCV | Image processing and webcam access |
| MediaPipe | Hand landmark detection |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| Scikit-learn | Label preprocessing |
| Flask | Web application development |
| HTML / CSS / JavaScript | User interface |

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) for hand gesture classification.

The model architecture includes:

1. Convolutional layer with 32 filters.
2. Max pooling layer.
3. Convolutional layer with 64 filters.
4. Max pooling layer.
5. Convolutional layer with 128 filters.
6. Flatten layer.
7. Fully connected Dense layer with 128 neurons.
8. Output layer with 24 neurons and Softmax activation.

**Model configuration:**
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 64
- Training Epochs: 10
- Input Image Size: 28 × 28 pixels
- Output Classes: 24

## 📂 Project Structure

```text
Sign-Language-Detection-System/
│
├── app.py
├── train_asl_model.py
├── sign_language_cnn_model.h5
│
├── templates/
│   └── index.html
│
├── static/
│   ├── styles.css
│   └── american_sign_language.PNG
│
├── README.md
└── requirements.txt
```

The training and testing CSV datasets are required to train the model and should be placed in the project root directory.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ranathi-2005/Sign-Language-Detection-System.git
```

### 2. Navigate to the project directory

```bash
cd Sign-Language-Detection-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install flask opencv-python numpy mediapipe tensorflow pandas scikit-learn
```

### 5. Run the application

Make sure the trained model file `sign_language_cnn_model.h5` is present in the project root directory.

```bash
python app.py
```

### 6. Open the application

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

Allow webcam access and show a hand gesture in front of the camera to see the system's prediction.

## 📊 Dataset

The project uses the Sign Language MNIST dataset, containing grayscale images of American Sign Language alphabet gestures.

- Image dimensions: 28 × 28 pixels.
- Training data: Sign MNIST training CSV.
- Testing data: Sign MNIST testing CSV.
- Classification categories: 24 letters.

The letters J and Z are excluded from the dataset because they involve motion.

The CSV datasets are not included in this repository and must be obtained separately for model retraining.

## 🚀 Future Improvements

- Improve prediction accuracy under different lighting conditions.
- Extend recognition to dynamic hand gestures.
- Support continuous sign language recognition.
- Convert recognized gestures into complete sentences.
- Add text-to-speech functionality.
- Improve the user interface and accessibility features.

## 👩‍💻 Author

**Ranathi**

GitHub: [@Ranathi-2005](https://github.com/Ranathi-2005)

## 📄 License

This project is available for educational and research purposes. Please check the dataset's original licensing terms before redistribution.
