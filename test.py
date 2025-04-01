# Raw Data Project Python File
# Authors:
# Alex Lester - 20411355
# George Choueiry - 20411131
# Jayaram Jeyakanthan 20410793

# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import joblib

# Function to preprocess data
def preprocess_data(df, activity_label, window):
    df.interpolate(method='linear', inplace=True)
    window_size = window
    df['x_filtered'] = df['Linear Acceleration x (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['y_filtered'] = df['Linear Acceleration y (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['z_filtered'] = df['Linear Acceleration z (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['Absolute_filtered'] = df['Absolute acceleration (m/s^2)'].rolling(window=window_size, center=True).mean()
    return df

def compare_accel_walk_jump(df_walk, df_jump, activity_label):
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    ax.plot(df_walk['Absolute_filtered'], label='Walk Absolute Acceleration')
    ax.plot(df_jump['Absolute_filtered'], label='Jump Absolute Acceleration')
    ax.set_title(f'{activity_label} - Absolute Acceleration')
    ax.set_xlabel('Time [s x 10^-2]')
    ax.set_ylabel('Acceleration [m/s²]')
    ax.legend()
    plt.show()

def extract_features(df, samples_per_window=500):
    features = []
    num_windows = len(df) // samples_per_window
    for i in range(num_windows):
        window = df['Absolute_filtered'].iloc[i*samples_per_window:(i+1)*samples_per_window].dropna()
        if len(window) == 0:
            continue
        f = [
            window.max(), window.min(), window.mean(), window.median(), window.std(),
            window.max() - window.min(), skew(window), kurtosis(window),
            np.sum(np.abs(window)), np.count_nonzero(np.diff(np.sign(window)))
        ]
        features.append(f)
    return np.array(features)

def normalize_features(features, scaler):
    return scaler.fit_transform(features)

def train_and_evaluate_model():
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    with h5py.File("data.h5", "r") as hdf:
        train_group = hdf["Segmented data/Train"]
        test_group = hdf["Segmented data/Test"]
        for key in train_group:
            if key.endswith(".X"):
                name = key[:-2]
                X_train_list.append(train_group[key][:])
                y_train_list.append(train_group[f"{name}.y"][:])
        for key in test_group:
            if key.endswith(".X"):
                name = key[:-2]
                X_test_list.append(test_group[key][:])
                y_test_list.append(test_group[f"{name}.y"][:])
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Walking", "Jumping"], yticklabels=["Walking", "Jumping"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
    joblib.dump(model, "logistic_model.pkl")

def predict_from_csv(input_path, output_path, window_size=500):
    model = joblib.load("logistic_model.pkl")
    df = pd.read_csv(input_path)
    df = preprocess_data(df, "Input Data", window=15)
    features = extract_features(df, samples_per_window=window_size)
    scaler = StandardScaler().fit(features)  # Refit scaler to input features
    features_normalized = normalize_features(features, scaler)
    predictions = model.predict(features_normalized)
    labels = ["Walking" if pred == 0 else "Jumping" for pred in predictions]
    output_df = pd.DataFrame({"Window": list(range(len(labels))), "Prediction": labels})
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def create_hdf5_file():
    with h5py.File("data.h5", "w") as hdf:
        raw_data_group = hdf.create_group("Raw data")
        pre_processed_group = hdf.create_group("Pre-processed data")
        segmented_data_group = hdf.create_group("Segmented data")
        raw_data_group.create_group("Alex")
        raw_data_group.create_group("George")
        raw_data_group.create_group("Jayaram")
        pre_processed_group.create_group("Alex")
        pre_processed_group.create_group("George")
        pre_processed_group.create_group("Jayaram")
        train_group = segmented_data_group.create_group("Train")
        test_group = segmented_data_group.create_group("Test")
        names = ["Alex", "George", "Jayaram"]
        types = ["Face", "FP", "Hand"]
        for i in range(0, len(names)):
            for k in range(0, len(types)):
                group_raw = hdf[f'Raw data/{names[i]}']
                group_process = hdf[f'Pre-processed data/{names[i]}']
                name = f'{names[i]}.{types[k]}'
                raw_walk_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Walking.csv')
                raw_jump_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Jumping.csv')
                process_walk_data = preprocess_data(raw_walk_data, f'{names[i]} {types[k]} Walking', 50)
                process_jump_data = preprocess_data(raw_jump_data, f'{names[i]} {types[k]} Jumping', 15)
                compare_accel_walk_jump(process_walk_data, process_jump_data, f'{names[i]} {types[k]}')
                group_raw.create_dataset(f'{name}.Walking', data=raw_walk_data.to_numpy())
                group_raw.create_dataset(f'{name}.Jumping', data=raw_jump_data.to_numpy())
                group_process.create_dataset(f'{name}.Walking', data=process_walk_data.to_numpy())
                group_process.create_dataset(f'{name}.Jumping', data=process_jump_data.to_numpy())
                features_walk = extract_features(process_walk_data)
                features_jump = extract_features(process_jump_data)
                features_combined = np.vstack((features_walk, features_jump))
                scaler = StandardScaler()
                features_normalized = normalize_features(features_combined, scaler)
                labels_walk = np.zeros(len(features_walk))
                labels_jump = np.ones(len(features_jump))
                labels = np.concatenate((labels_walk, labels_jump))
                indices = np.random.permutation(len(labels))
                X = features_normalized[indices]
                y = labels[indices]
                split_idx = int(0.9 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                hdf["Segmented data/Train"].create_dataset(f'{name}.X', data=X_train)
                hdf["Segmented data/Train"].create_dataset(f'{name}.y', data=y_train)
                hdf["Segmented data/Test"].create_dataset(f'{name}.X', data=X_test)
                hdf["Segmented data/Test"].create_dataset(f'{name}.y', data=y_test)

if __name__ == '__main__':
    create_hdf5_file()
    train_and_evaluate_model()
    predict_from_csv("data_test1.csv", "predictions.csv")



