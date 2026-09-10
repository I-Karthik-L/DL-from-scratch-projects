# Brain Tumor Classification with CNNs — A Learning Journey

This repository documents my journey learning Convolutional Neural Networks (CNNs) by building a brain tumor MRI classifier in three progressive stages: a CNN from scratch, the same model with data augmentation, and finally transfer learning with a pretrained ResNet50. Each notebook builds on the last, so you can follow the same path I did.

## 📁 Repository Structure

```
cnn/
├── 01_basic_cnn.ipynb              # CNN built from scratch
├── 02_image_augmentation.ipynb     # + on-the-fly data augmentation
├── 03_transfer_learning.ipynb      # + ResNet50 transfer learning
└── README.md
```



## 🧠 What's in Each Notebook

| # | Notebook | Concepts Covered |
|---|----------|-------------------|
| 1 | **Basic CNN** | Loading & preprocessing images with OpenCV, grayscale conversion, resizing, normalization, one-hot encoding, building a `Conv2D → ReLU → MaxPooling` stack from scratch, `Dropout` for regularization, training and evaluating a classifier |
| 2 | **Image Augmentation** | Keras preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomTranslation`, `RandomZoom`, `RandomContrast`) applied as model layers (GPU-accelerated, training-only), `EarlyStopping` with `restore_best_weights` |
| 3 | **Transfer Learning** | Using a pretrained **ResNet50** (ImageNet weights) as a frozen feature extractor, switching from grayscale to RGB input, `GlobalAveragePooling2D` instead of `Flatten`, combining a pretrained backbone with a custom classification head |

## 📊 Dataset

Brain MRI scans organized into two categories (tumor / no tumor), loaded from Google Drive in the format:
```
Brain_Tumor/
├── yes/
└── no/
```
*(Data_set is provided as zip file in the repository,you can download it.)*

## 🚀 Getting Started

These notebooks were built and run in **Google Colab**. To use them:

1. Upload the dataset to your Google Drive, keeping the `yes/` / `no/` folder structure above.
2. Open a notebook in Colab and mount your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update `data_path` to point to your dataset location.
4. Run the cells top to bottom.

### Requirements
```
tensorflow>=2.x
opencv-python
numpy
matplotlib
scikit-learn
```

## 🛠️ Tech Stack
- **TensorFlow / Keras** — model building and training
- **OpenCV** — image loading and preprocessing
- **scikit-learn** — train/test splitting
- **Matplotlib** — visualization

## 📈 Results

| Notebook | Test Accuracy |
|----------|:---:|
| Basic CNN | 76% |
| + Augmentation | 80% |
| + Transfer Learning (ResNet50) | 88% |


## 🙏 Acknowledgments

Built as a hands-on introduction to CNNs and transfer learning for image classification.


