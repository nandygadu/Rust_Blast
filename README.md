# Crop Disease Detection Web Application (Blast vs. Rust)

## 📌 Project Overview
This repository contains a complete, end-to-end computer vision web application designed to aid in digital agricultural diagnostics. Using a custom-trained **Convolutional Neural Network (CNN)** built with TensorFlow and Keras, the system analyzes digital images of crop leaves to accurately classify and detect the presence of destructive fungal infections, specifically **Blast** and **Rust** diseases.

The model is deployed via a lightweight **Flask backend**, allowing farmers, agronomists, or researchers to upload leaf images through an intuitive web interface, process them instantly, and view the diagnostic classification alongside a model confidence probability score.

---

## 🚀 Features
* **Automated Image Diagnostics:** Instantly classifies crop leaf images into specific disease categories (Blast or Rust).
* **Deep Learning Inference:** Backed by an optimized Convolutional Neural Network (CNN) trained with Keras/TensorFlow.
* **Real-time Preprocessing:** Automatically resizes ($224 \times 224$), normalizes, and expands array dimensions of incoming images on the fly.
* **User-Friendly Web Interface:** Simple file-upload mechanics with structured, dynamic result rendering.
* **Secure File Handling:** Implements secure naming protocols for uploaded files before processing.

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, Jinja2 Templates
* **Backend Framework:** Flask (Python)
* **Machine Learning & CV:** TensorFlow, Keras, NumPy
* **Image Processing:** Pillow (PIL)

---

## 📁 Repository Structure
```text
├── bert/                      # (Optional) Related classification configurations
├── static/                    # CSS stylesheets and UI images
├── templates/                 # HTML templates
│   ├── index.html             # Main dashboard file upload page
│   └── result.html            # Diagnostic results displaying metrics
├── uploads/                   # Temporary server storage for uploaded images
├── app.py                     # Main Flask web application backend router
├── Blast_Rust_Model.keras     # Pre-trained CNN model weights
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
