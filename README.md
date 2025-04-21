# Human Activity Classifier (ELEC 292 Project)

This project is a machine learning pipeline designed to classify human activities—specifically distinguishing between **walking** and **jumping**—based on accelerometer data collected from mobile sensors. It uses **logistic regression** to train a classifier on preprocessed, feature-extracted data and includes a **Tkinter GUI app** for uploading new data and visualizing predictions.

## 📁 Project Structure

├── Raw Data/ # Original CSV files (accelerometer data) ├── data.h5 # HDF5 file storing raw, preprocessed, and segmented data ├── logistic_model.pkl # Trained logistic regression model ├── new_data.csv # Example input file for prediction ├── new_data_labeled.csv # Example output file with predicted labels ├── main.py # Main pipeline and GUI code └── README.md # This file


---

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

