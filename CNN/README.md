# Brain Tumor Detection CNN

This repository contains an end-to-end deep learning workflow for detecting brain tumors from MRI scans. The project covers everything from raw data preprocessing and Exploratory Data Analysis (EDA) to building and structuring a Convolutional Neural Network (CNN) using TensorFlow/Keras.

## Project Overview
The primary objective of this notebook is to classify MRI images into two distinct categories:
*   **Yes:** MRI scans showing the presence of a brain tumor.
*   **No:** Healthy MRI scans with no tumor.

## Workflow & Architecture
The Jupyter Notebook is structured into clear phases to demonstrate a complete computer vision pipeline:

### 1. Data Preprocessing & EDA (OpenCV)
*   **Dynamic Labeling:** Automatically maps directory names to binary numerical labels (`0` and `1`).
*   **Grayscale Conversion:** Converts raw RGB images into grayscale to reduce computational load and isolate structural features.
*   **Image Resizing:** Standardizes all input images to a uniform `100x100` pixel dimension.
*   **Fault Tolerance:** Implements robust `try...except` blocks to bypass unreadable or corrupted files in the dataset.
*   **Visualization:** Utilizes Matplotlib to display processed image arrays.

### 2. Convolutional Neural Network (TensorFlow / Keras)
The notebook constructs a `Sequential` CNN model designed to extract spatial hierarchies and classify the scans:
*   **Convolutional Blocks (`Conv2D`):** Successive layers of 200 and 100 filters (using `3x3` kernels) to detect edges, shapes, and complex patterns.
*   **Activation:** Applies `ReLU` activation functions to introduce non-linearity.
*   **Pooling (`MaxPooling2D`):** Uses `2x2` pooling windows to downsample feature maps and reduce dimensionality.
*   **Regularization (`Dropout`):** Applies a 50% dropout rate before the dense layers to prevent the model from overfitting the training data.
*   **Classification Head (`Dense`):** A hidden layer with 50 neurons feeds into a final 2-neuron output layer using a `softmax` activation to output a clear probability distribution.

## How to Run This Code (Google Colab)
This project is configured to run easily in **Google Colab** using data hosted on **Google Drive**.

1. **Download the Data:** Download the dataset `.zip` file provided in this repository and extract it to your local machine.
2. **Upload to Google Drive:** Upload the extracted folder (ensure it is named `Brain_Tumor`) directly to the root of your Google Drive (`MyDrive`).
3. **Open the Notebook:** Upload the `.ipynb` file to Google Colab.
4. **Mount Your Drive:** Run the following snippet in a new cell at the top of the notebook to grant Colab access to your images:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
