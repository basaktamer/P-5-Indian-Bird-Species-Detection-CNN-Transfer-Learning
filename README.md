# Indian Bird Species Detection using Deep CNN & Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-info.svg)](https://www.kaggle.com/code/basaktamer/indian-bird-species-detection-cnn-tl)

## 📌 Project Overview
India is home to a diverse range of avian species, many of which look remarkably similar. This project focuses on building a robust **Deep Learning** pipeline to automatically identify 25 different Indian bird species from a dataset of over 22,000 images. The goal is to create a model capable of distinguishing subtle visual markers such as feather patterns, beak structures, and color gradients.

## 📊 Dataset Information
- **Source:** [25 Indian Bird Species - Kaggle](https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images)
- **Scale:** ~22,600 images.
- **Classes:** 25 unique species native to the Indian subcontinent.

## 🛠️ Technical Workflow
1. **Data Engineering:** - Preprocessed high-resolution images for consistent input size.
   - Applied **Data Augmentation** (Zoom, Horizontal Flip, Rotation) to help the model generalize across different lighting conditions and bird postures.
2. **Architecture Strategy:**
   - **Deep Custom CNN:** Designed a specialized architecture with **5 Convolutional Layers** and **3 Pooling Layers** to learn hierarchical features from scratch.
   - **Regularization:** Integrated **Dropout layers** and **Batch Normalization** to prevent overfitting and ensure stable convergence.
   - **Transfer Learning:** Fine-tuned pre-trained models like **MobileNet** and **VGG16** to compare performance against the custom baseline.
3. **Evaluation:**
   - Monitored accuracy and loss curves to optimize hyper-parameters.
   - Used a **Classification Report** to analyze precision and recall across all 25 species.

## 🚀 Key Insights
- **Feature Complexity:** The 5-layer CNN architecture successfully captured fine-grained details necessary for species identification.
- **Efficiency:** MobileNet was explored as a lightweight alternative, demonstrating potential for mobile-based bird-watching applications.

## 📂 Repository Structure
- `Indian_Bird_Species_Detection.ipynb`: Full training and evaluation pipeline.
- `README.md`: Project documentation.

---
**Author:** Basak Tamer  
**Part of:** 41-Project Data Science Portfolio