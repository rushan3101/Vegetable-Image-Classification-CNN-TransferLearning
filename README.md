# 🥦 Vegetable Image Detection using Deep Learning

### 🔗 [Try the Live App](https://vegetable-image-classification-rushan3101.streamlit.app/)

## 📘 Project Overview

This project is an end-to-end **Deep Learning-based image classification system** for identifying different types of vegetables from images. It started as a model comparison experiment using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**, and has since grown into a full mobile-friendly app: users can scan or upload a photo of a vegetable, get it classified in real time, and ask follow-up questions about it (nutrition, storage, cooking tips) through an integrated LLM assistant.

---

## 🎯 Objectives

* Classify vegetable images accurately using deep learning models.
* Experiment with custom CNN architectures and transfer learning to improve performance.
* Optimize the model through fine-tuning to achieve near-perfect accuracy.
* Package the best model into a real, usable mobile app rather than a notebook-only result.

---

## 🧠 Approach

1. **Data Preprocessing**
   * Resized and normalized all input images for consistency.

2. **Model Development**
   * Built a baseline **CNN model** achieving ~94% accuracy.
   * Implemented **Transfer Learning** using multiple pre-trained architectures (VGG19, MobileNetV2, EfficientNetB0).
   * Fine-tuned layers and hyperparameters for better feature extraction and generalization.

3. **Model Evaluation**
   * Achieved **99.9% accuracy** with **Fine-Tuned EfficientNetB0**, outperforming other architectures.
   * Evaluated using accuracy, loss curves, per-class precision/recall/F1, and a confusion matrix (see [Evaluation](#-evaluation) below).

4. **Deployment**
   * Exported the fine-tuned EfficientNetB0 model (`.keras`) along with its class-index mapping.
   * Built a **Streamlit app** with a mobile-optimized camera/upload interface for live classification.
   * Integrated **Gemini 3.1 Flash-Lite via LangChain** so users can ask natural-language questions about the identified vegetable, grounded in a curated fact-sheet to reduce hallucination.

---

## 🧩 Tech Stack

* **Language:** Python
* **Modeling:** TensorFlow, Keras, NumPy, Matplotlib, scikit-learn
* **Model Architectures:** Custom CNN, VGG19, MobileNetV2, EfficientNetB0
* **App:** Streamlit (mobile-optimized UI, camera input)
* **LLM Integration:** LangChain, Google Gemini 3.1 Flash-Lite
* **Tools:** Jupyter Notebook, Google Colab, VS Code

---

## 📊 Model Comparison

| Model                        | Accuracy  |
| ----------------------------- | --------- |
| Custom CNN                    | 94.4%     |
| MobileNetV2                   | 99.6%     |
| MobileNetV2 (Fine Tuned)      | 99.8%     |
| VGG19                         | 99.3%     |
| VGG19 (Fine Tuned)            | 99.6%     |
| EfficientNetB0                | 99.8%     |
| **EfficientNetB0 (Fine Tuned)** | **99.9%** |

The fine-tuned EfficientNetB0 model was selected as the final model powering the app.

---

## 📈 Evaluation

### Classification Report (EfficientNetB0, Fine-Tuned)

| Vegetable      | Precision | Recall | F1-score | Samples |
| -------------- | --------- | ------ | -------- | ------- |
| Bean           | 1.00      | 1.00   | 1.00     | 200     |
| Bitter Gourd   | 1.00      | 0.99   | 0.99     | 200     |
| Bottle Gourd   | 1.00      | 1.00   | 1.00     | 200     |
| Brinjal        | 1.00      | 1.00   | 1.00     | 200     |
| Broccoli       | 1.00      | 1.00   | 1.00     | 200     |
| Cabbage        | 1.00      | 0.99   | 0.99     | 200     |
| Capsicum       | 1.00      | 1.00   | 1.00     | 200     |
| Carrot         | 1.00      | 1.00   | 1.00     | 200     |
| Cauliflower    | 1.00      | 1.00   | 0.99     | 200     |
| Cucumber       | 1.00      | 1.00   | 0.99     | 200     |
| Papaya         | 1.00      | 1.00   | 1.00     | 200     |
| Potato         | 1.00      | 1.00   | 1.00     | 200     |
| Pumpkin        | 1.00      | 1.00   | 1.00     | 200     |
| Radish         | 1.00      | 1.00   | 1.00     | 200     |
| Tomato         | 1.00      | 1.00   | 1.00     | 200     |

### Confusion Matrix

![Confusion Matrix](metrics/Confusion_Matrix.png)

Out of 3,000 test images (200 per class), only 2 were misclassified — one `Bitter_Gourd` predicted as `Cucumber`, and one `Cabbage` predicted as `Cucumber`.

---

## 📱 The App

The fine-tuned model is deployed behind a mobile-friendly Streamlit interface:

1. **Scan or upload** a photo of a vegetable using your phone camera or a file.
2. **Get an instant prediction** with confidence score and top-3 alternatives.
3. **Ask questions** about the identified vegetable — nutrition, storage, recipes, selection tips — answered by Gemini 3.1 Flash-Lite via LangChain, grounded in a curated reference fact-sheet for each class.

**[🔗 Live App](https://vegetable-image-classification-rushan3101.streamlit.app/)**

---

## 📂 Project Structure

```
Vegetable-Image-Classification-CNN-TransferLearning-EfficientNetB0-LLM-Gemini-Streamlit/
├── Vegetable.ipynb              # model training & 
├── app/
│   ├── streamlit_app.py         # main app / UI
│   ├── llm_chain.py             # LangChain + Gemini 3.1 Flash-Lite
│   ├── utils.py                 # model loading, preprocessing, prediction
│   └── vegetable_facts.json     # curated per-class knowledge base
├── models/
│   ├── eff_model.keras          # fine-tuned EfficientNetB0
│   └── class_indices.json       # class name <-> index mapping
├── metrics/
│   ├── Confusion_Matrix.png
│   └── classification_report.csv
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/rushan3101/Vegetable-Image-Classification-CNN-TransferLearning-EfficientNetB0-LLM-Gemini-Streamlit.git
cd Vegetable-Image-Classification-CNN-TransferLearning-EfficientNetB0-LLM-Gemini-Streamlit

# 2. Install dependencies
pip install uv
uv sync

# 3. Add your Gemini API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your key

# 4. Run the app
streamlit run app/streamlit_app.py
```

Model training and experimentation can be found in `notebooks/Vegetable.ipynb`.

---

## 🚀 Key Features

* End-to-end deep learning workflow (data preprocessing → model training → evaluation → deployment)
* Transfer learning comparison across multiple pre-trained architectures
* 99.9% test accuracy with detailed per-class evaluation (precision, recall, F1, confusion matrix)
* Mobile-optimized live camera classification via Streamlit
* LLM-powered Q&A (LangChain + Gemini 3.1 Flash-Lite) grounded in curated vegetable facts

---

## Limitation

The model is trained on only 15 vegetable classes (Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato). Any vegetable, fruit, or object outside this list will be misclassified as the closest-looking known class, or flagged as low-confidence if it falls below the app's confidence threshold.


## 🏁 Conclusion

This project demonstrates the power of **Transfer Learning** and **Fine-Tuning** in achieving near-perfect performance on an image classification problem, and takes it a step further by turning the model into a usable, mobile-friendly product with an integrated LLM assistant. It can be extended for broader agricultural or retail use cases involving image-based product identification and consumer education.