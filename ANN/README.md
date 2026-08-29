# Artificial Neural Network (ANN) - MNIST Digit Classification

This project demonstrates how to build and train an **Artificial Neural Network (ANN)** to classify handwritten digits using the **MNIST dataset**.

The goal of this project is to understand the fundamental workflow of a Deep Learning project—from loading and preprocessing data to building, training, and evaluating a neural network.

## 📌 Project Overview

The MNIST dataset contains handwritten digit images ranging from **0 to 9**. Each image is a **28 × 28 grayscale image**.

In this project, the images are processed and passed through an Artificial Neural Network to predict the corresponding digit.

## 🧠 Model Architecture

The neural network consists of:

* **Flatten Layer** – Converts the 28 × 28 image into a one-dimensional vector.
* **Dense Layer (128 neurons)** – Uses the ReLU activation function to learn patterns from the input.
* **Dropout Layer (0.2)** – Helps reduce overfitting during training.
* **Output Layer (10 neurons)** – Uses the Softmax activation function to predict probabilities for digits 0–9.

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

## 🚀 Workflow

1. Load the MNIST dataset
2. Normalize pixel values to a range of 0 to 1
3. Visualize sample images
4. Build the ANN model
5. Compile the model using the Adam optimizer and Sparse Categorical Crossentropy loss
6. Train the model for 10 epochs
7. Visualize training and validation loss
8. Make predictions on the test dataset
9. Evaluate the model's test accuracy

## 📊 Key Concepts Covered

* Artificial Neural Networks (ANN)
* Data preprocessing and normalization
* Dense layers
* ReLU and Softmax activation functions
* Dropout and overfitting
* Model training and validation
* Loss functions and optimizers
* Making predictions and evaluating accuracy

## ▶️ How to Run

Clone this repository:

```bash
git clone https://github.com/I-Karthik-L/DL-from-scratch-projects.git
```

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib
```

Then open and run the `ANN.ipynb` notebook using Jupyter Notebook or Google Colab.

## 🎯 Learning Goal

This project is part of my **Deep Learning journey**, where I build projects from scratch to better understand the core concepts of Deep Learning through hands-on implementation.

Feel free to explore the notebook and use it as a learning resource!
