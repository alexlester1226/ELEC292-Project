# Human Activity Classifier (ELEC 292 Project)

This project is a machine learning pipeline designed to classify human activities—specifically distinguishing between **walking** and **jumping**—based on accelerometer data collected from mobile sensors. It uses **logistic regression** to train a classifier on preprocessed, feature-extracted data and includes a **Tkinter GUI app** for uploading new data and visualizing predictions.


## 🔍 Features

- **Preprocessing**: Linear interpolation to handle missing values and Simple Moving Average (SMA) filters for smoothing.
- **Feature Extraction**: Extracts statistical features from windows of absolute acceleration.
- **Train/Test Splitting**: Automatically segments and labels windows with balanced training and test sets.
- **Model Training**: Logistic regression pipeline using scikit-learn.
- **Evaluation**: Outputs accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.
- **GUI**: A simple desktop app (built with Tkinter) to upload a CSV, make predictions, and visualize results.

---

## 📊 Feature Extraction Metrics

Each window of accelerometer data is processed into a 10-dimensional feature vector:

- Max
- Min
- Mean
- Median
- Standard deviation
- Range (max - min)
- Skewness
- Kurtosis
- Sum of absolute values
- Number of zero crossings

---

