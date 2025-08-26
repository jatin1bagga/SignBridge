
# 🧤 Glove Gesture Recognition with 1D CNN

## 📌 Project Overview

This project implements a **glove-based gesture recognition system** using flex sensors and accelerometer data.
The data is processed using a **1D Convolutional Neural Network (1D CNN)** in **TensorFlow/Keras** to classify gestures into 9 categories:

* **BEAUTIFUL**
* **CAMERA**
* **CAR**
* **DARK**
* **PAPER**
* **SWEET**
* **TABLE**
* **TEACHER**
* **TREE**

The system is deployed on the **cloud backend (Flask API on Google Cloud)** so that sensor data can be sent from the ESP32 glove, processed, and the predicted gesture retrieved by a mobile app in real time.

---

## ⚙️ Features

* Input features:
  `Xl, Yl, Zl, Flex1, Flex2, Flex3, Flex5, Xr, Yr, Zr, Flex6, Flex7, Flex8, Flex9, Flex10`
* Model: **1D CNN** with batch normalization, dropout, and Adam optimizer.
* Accuracy: \~97% on validation data.
* REST API endpoints:

  * `POST /predict` → send sensor data, get prediction.
  * `GET /get-prediction` → fetch latest stored prediction.
* Mobile app (Expo/React Native) fetches prediction and displays real-time gesture.

---

## 🛠️ Tech Stack

* **Hardware:** ESP32 + Flex sensors + ADXL accelerometer
* **Backend:** Flask (Python), TensorFlow/Keras
* **ML Model:** 1D CNN with softmax classification
* **Deployment:** Google Cloud App Engine
* **Frontend:** React Native (Expo) mobile app

---

## 🚀 How It Works

1. **ESP32 Glove** collects sensor data.
2. Data sent to **Flask API** (`/predict`).
3. Flask backend loads **trained CNN model (.h5)**, **scaler**, and **label encoder**.
4. Prediction is computed and stored.
5. Mobile app queries `/get-prediction` every few seconds to display gesture.

---

## 🔗 API Usage

### 1. POST `/predict`

Send glove sensor data (example JSON):

```json
{
  "samples": [
    {
      "Xl": 2.51, "Yl": 3.41, "Zl": 9.3,
      "Flex1": 1158, "Flex2": 1275, "Flex3": 125, "Flex5": 1390,
      "Xr": -5.88, "Yr": 0.12, "Zr": 10.2,
      "Flex6": 1008, "Flex7": 1327, "Flex8": 1683, "Flex9": 1109, "Flex10": 889
    }
  ]
}
```

Response:

```json
{
  "status": "ok",
  "majority_gesture": "TABLE",
  "confidence_hint": 0.95
}
```

### 2. GET `/get-prediction`

Fetch latest prediction:

```json
{
  "gesture": "TABLE",
  "prob": 0.95
}
```

---

## 📊 Model Training

* Preprocessing: StandardScaler for feature normalization, LabelEncoder for labels.
* Loss: `categorical_crossentropy`
* Optimizer: **Adam (lr=0.001)**
* Batch size: 32
* Epochs: up to 100 (with **EarlyStopping** + **ModelCheckpoint**)
* Achieved \~97% validation accuracy

---

## 📦 Files in Repo

* `Data.csv` → Training dataset
* `train_cnn.py` → Training script
* `best_model.h5` → Saved trained model
* `scaler.pkl`, `label_encoder.pkl` → Preprocessing artifacts
* `app.py` → Flask backend (cloud API)
* `mobile_app/` → React Native frontend (optional)

---

## 📈 Results

* High accuracy gesture recognition
* Robust to noise (dropout + batch norm applied)
* Real-time cloud deployment

---

## ✅ Next Steps

* Optimize for faster inference on edge devices.
* Add more gestures to dataset.
* Explore hybrid models (CNN + LSTM) for sequential data.

---

## 👨‍💻 Author

Jatin Bagga

---
