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
def preprocess_data(df, window):
    # fills in any missing values (NaNs) in the dataset using linear interpolation.
    df.interpolate(method='linear', inplace=True)

    # Apply SMA filter on x, y, z and absolute acceleration data
    df['x_filtered'] = df['Linear Acceleration x (m/s^2)'].rolling(window=window, center=True).mean()
    df['y_filtered'] = df['Linear Acceleration y (m/s^2)'].rolling(window=window, center=True).mean()
    df['z_filtered'] = df['Linear Acceleration z (m/s^2)'].rolling(window=window, center=True).mean()
    df['Absolute_filtered'] = df['Absolute acceleration (m/s^2)'].rolling(window=window, center=True).mean()

    return df


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

# Function to train and evaluate the model
def train_and_evaluate_model():
    # Lists to store training and testing data from all users/sensors
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    # Open the HDF5 file and access training and testing datasets
    with h5py.File("data.h5", "r") as hdf:
        train_group = hdf["Segmented data/Train"]
        test_group = hdf["Segmented data/Test"]

        # Collect training data and labels from all subgroups
        for key in train_group:
            if key.endswith(".X"):
                name = key[:-2]
                X_train_list.append(train_group[key][:])
                y_train_list.append(train_group[f"{name}.y"][:])

        # Collect testing data and labels from all subgroups
        for key in test_group:
            if key.endswith(".X"):
                name = key[:-2]
                X_test_list.append(test_group[key][:])
                y_test_list.append(test_group[f"{name}.y"][:])

    # Combine data from all users/sensors into single training and testing sets
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict class labels and probabilities on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    conf_mat = confusion_matrix(y_test, y_pred)

    # Print the evaluation results
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Walking", "Jumping"], yticklabels=["Walking", "Jumping"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Plot the ROC curve
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

    # Save the trained model to a file for future use
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


def prepare_train_test_data(features_walk, features_jump):
    # Combine features
    features_combined = np.vstack((features_walk, features_jump))

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_combined)

    # Assign labels (0 for walk, 1 for jump)
    labels_walk = np.zeros(len(features_walk))
    labels_jump = np.ones(len(features_jump))
    labels = np.concatenate((labels_walk, labels_jump))

    # Shuffle data
    indices = np.random.permutation(len(labels))
    X = features_normalized[indices]
    y = labels[indices]

    # Split data into train and test sets (90:10 split)
    split_idx = int(0.9 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def create_hdf5_file():
    # Open in write mode the "data.h5" HDF5 file
    with h5py.File("data.h5", "w") as hdf:

        # create 3 main groups
        raw_data_group = hdf.create_group("Raw data")
        pre_processed_group = hdf.create_group("Pre-processed data")
        segmented_data_group = hdf.create_group("Segmented data")

        # create subgroups within main groups
        raw_data_group.create_group("Alex")
        raw_data_group.create_group("George")
        raw_data_group.create_group("Jayaram")
        pre_processed_group.create_group("Alex")
        pre_processed_group.create_group("George")
        pre_processed_group.create_group("Jayaram")
        train_group = segmented_data_group.create_group("Train")
        test_group = segmented_data_group.create_group("Test")

        # Arrays of strings for filenames
        names = ["Alex", "George", "Jayaram"]
        types = ["Face", "FP", "Hand"]

        # double for loop to iterate through both arrays
        for i in range(0, len(names)):
            for k in range(0, len(types)):
                # select created groups from HDF5 file
                group_raw = hdf[f'Raw data/{names[i]}']
                group_process = hdf[f'Pre-processed data/{names[i]}']

                # read the csv files using pandas
                name = f'{names[i]}.{types[k]}'
                raw_walk_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Walking.csv')
                raw_jump_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Jumping.csv')

                # create the raw data datasets
                group_raw.create_dataset(f'{name}.Walking', data=raw_walk_data.to_numpy())
                group_raw.create_dataset(f'{name}.Jumping', data=raw_jump_data.to_numpy())

                # call the pre-process data function
                process_walk_data = preprocess_data(raw_walk_data, 50)
                process_jump_data = preprocess_data(raw_jump_data, 15)

                # create the pre-processed datasets
                group_process.create_dataset(f'{name}.Walking', data=process_walk_data.to_numpy())
                group_process.create_dataset(f'{name}.Jumping', data=process_jump_data.to_numpy())

                # call the extract_features function and prepare_train_test_data function
                features_walk = extract_features(process_walk_data)
                features_jump = extract_features(process_jump_data)
                X_train, X_test, y_train, y_test = prepare_train_test_data(features_walk, features_jump)

                # create the Train and Test datasets
                hdf["Segmented data/Train"].create_dataset(f'{name}.X', data=X_train)
                hdf["Segmented data/Train"].create_dataset(f'{name}.y', data=y_train)
                hdf["Segmented data/Test"].create_dataset(f'{name}.X', data=X_test)
                hdf["Segmented data/Test"].create_dataset(f'{name}.y', data=y_test)

                # call function to graph raw and processed data
                graph_data(process_walk_data, process_jump_data, f'{names[i]} {types[k]}')

def xyz_raw_filtered_graph(df, activity_label):
    # PLOT 1: X, Y, Z Raw vs Filtered (Separate Plots)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{activity_label} - Raw vs Filtered Accelerometer Data (X, Y, Z)')

    # Top: Raw x, y, z acceleration
    axs[0].plot(df['Linear Acceleration x (m/s^2)'], label='Raw X')
    axs[0].plot(df['Linear Acceleration y (m/s^2)'], label='Raw Y')
    axs[0].plot(df['Linear Acceleration z (m/s^2)'], label='Raw Z')
    axs[0].set_title('Raw Acceleration')
    axs[0].legend()

    # Bottom: Filtered x, y, z acceleration
    axs[1].plot(df['x_filtered'], label='Filtered X')
    axs[1].plot(df['y_filtered'], label='Filtered Y')
    axs[1].plot(df['z_filtered'], label='Filtered Z')
    axs[1].set_title('Filtered Acceleration (MA filter)')
    axs[1].legend()

    plt.xlabel('Time [s x 10^-2]')
    plt.ylabel('Acceleration [m/s²]')
    plt.tight_layout()
    plt.show()
def absolute_raw_filtered_graph(df, activity_label):
    # Absolute Acceleration - Raw vs Filtered (Separate Plots)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{activity_label} - Absolute Acceleration')

    # Top: Raw absolute acceleration
    axs[0].plot(df['Absolute acceleration (m/s^2)'], label='Raw Absolute Acceleration')
    axs[0].set_title('Raw Absolute Acceleration')
    axs[0].set_ylabel('Acceleration [m/s²]')
    axs[0].legend()

    # Bottom: Filtered absolute acceleration
    axs[1].plot(df['Absolute_filtered'], label='Filtered Absolute Acceleration')
    axs[1].set_title('Filtered Absolute Acceleration (MA Filter)')
    axs[1].set_xlabel('Time [s x 10^-2]')
    axs[1].set_ylabel('Acceleration [m/s²]')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def compare_accel_walk_jump(df_walk, df_jump, activity_label):
    # Filtered Absolute Acceleration - Walking vs Jumping
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    ax.plot(df_walk['Absolute_filtered'], label='Walk Absolute Acceleration')
    ax.plot(df_jump['Absolute_filtered'], label='Jump Absolute Acceleration')
    ax.set_title(f'{activity_label} - Absolute Acceleration')
    ax.set_xlabel('Time [s x 10^-2]')
    ax.set_ylabel('Acceleration [m/s²]')
    ax.legend()
    plt.show()

def graph_data(process_walk_data, process_jump_data, label):
    # function that graphs the raw versus processed x, y, z acceleration for both walking and jumping
    xyz_raw_filtered_graph(process_walk_data, f'{label} Walking')
    xyz_raw_filtered_graph(process_jump_data, f'{label} Jumping')

    # function that graphs the raw versus processed absolute acceleration for both walking and jumping
    absolute_raw_filtered_graph(process_walk_data, f'{label} Walking')
    absolute_raw_filtered_graph(process_jump_data, f'{label} Jumping')

    # function that graphs the processed absolute acceleration of walking versus jumping
    compare_accel_walk_jump(process_walk_data, process_jump_data, label)


# main function
if __name__ == '__main__':
    create_hdf5_file()
    train_and_evaluate_model()
    predict_from_csv("ELEC252.csv", "predictions.csv")




