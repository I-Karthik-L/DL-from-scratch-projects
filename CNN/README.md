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
*(Add a link to the exact dataset you used — e.g. the Kaggle "Brain MRI Images for Brain Tumor Detection" dataset — so others can reproduce this.)*

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
| Basic CNN | _fill in_ |
| + Augmentation | _fill in_ |
| + Transfer Learning (ResNet50) | _fill in_ |

*(Fill these in from your notebook outputs — the accuracy generally improves as augmentation and transfer learning are added.)*

## 🔍 Notes, Corrections & Future Improvements

Being transparent about what I'd change if I revisited this:

- **Validation/test overlap:** across all three notebooks, the same held-out set is used both as the training-time validation set (including for `EarlyStopping`'s best-weights selection) *and* as the final test set. This can inflate the reported test accuracy slightly, since the model selection is implicitly informed by that same data. A proper 3-way split (train / validation / test) would give a more honest number.
- **Missing `None` checks on image loads:** `cv2.imread()` returns `None` for unreadable files, which is currently only caught indirectly via a generic `try/except`. Explicit `if img is None: continue` with a counter would make data-quality issues visible.
- **ResNet50 preprocessing:** notebook 3 normalizes with plain `/255.0`, but ResNet50 was trained with `tensorflow.keras.applications.resnet50.preprocess_input`, which does mean-subtraction rather than simple scaling. Switching to it would better match the pretrained weights.
- **Input resolution for transfer learning:** images are resized to 100×100, well below ResNet50's native 224×224 — it still runs (thanks to global average pooling), but a larger input size would likely help.
- **No fine-tuning stage:** notebook 3 keeps the ResNet50 base fully frozen. A natural next step is unfreezing the top few blocks and fine-tuning at a low learning rate.
- **No class-balance check or confusion matrix:** for a medical-imaging task, it's worth confirming the two classes are reasonably balanced and looking at precision/recall, not just accuracy.

## 🙏 Acknowledgments

Built as a hands-on introduction to CNNs and transfer learning for image classification.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
