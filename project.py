# # Raw Data Project Python File
# # Authors:
# # Alex Lester - 20411355
# # George Choueiry - 20411131
# # Jayaram Jeyakanthan 20410793
#
# # import statements
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# from scipy.stats import skew, kurtosis
# from sklearn.preprocessing import StandardScaler
#
#
#
# # Function to preprocess data
# def preprocess_data(df, activity_label, window):
#     # Interpolate missing values
#     df.interpolate(method='linear', inplace=True)
#
#     window_size = window  # Window size for MA filter
#
#     # Apply MA filter to linear accelerations
#     df['x_filtered'] = df['Linear Acceleration x (m/s^2)'].rolling(window=window_size, center=True).mean()
#     df['y_filtered'] = df['Linear Acceleration y (m/s^2)'].rolling(window=window_size, center=True).mean()
#     df['z_filtered'] = df['Linear Acceleration z (m/s^2)'].rolling(window=window_size, center=True).mean()
#
#     # Apply MA filter to existing absolute acceleration column
#     df['Absolute_filtered'] = df['Absolute acceleration (m/s^2)'].rolling(window=window_size, center=True).mean()
#
#     # # === PLOT 1: X, Y, Z Raw vs Filtered ===
#     # fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     # fig.suptitle(f'{activity_label} - Raw vs Filtered Accelerometer Data (X, Y, Z)')
#     #
#     # axs[0].plot(df['Linear Acceleration x (m/s^2)'], label='Raw X')
#     # axs[0].plot(df['Linear Acceleration y (m/s^2)'], label='Raw Y')
#     # axs[0].plot(df['Linear Acceleration z (m/s^2)'], label='Raw Z')
#     # axs[0].set_title('Raw Acceleration')
#     # axs[0].legend()
#     #
#     # axs[1].plot(df['x_filtered'], label='Filtered X')
#     # axs[1].plot(df['y_filtered'], label='Filtered Y')
#     # axs[1].plot(df['z_filtered'], label='Filtered Z')
#     # axs[1].set_title('Filtered Acceleration (MA filter)')
#     # axs[1].legend()
#     #
#     # plt.xlabel('Time [s x 10^-2]')
#     # plt.ylabel('Acceleration [m/s²]')
#     # plt.tight_layout()
#     # plt.show()
#     #
#     # # === PLOT 2: Absolute Acceleration - Raw vs Filtered (Separate Plots) ===
#     # fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     # fig.suptitle(f'{activity_label} - Absolute Acceleration')
#     #
#     # # Top: Raw absolute acceleration
#     # axs[0].plot(df['Absolute acceleration (m/s^2)'], label='Raw Absolute Acceleration')
#     # axs[0].set_title('Raw Absolute Acceleration')
#     # axs[0].set_ylabel('Acceleration [m/s²]')
#     # axs[0].legend()
#     #
#     # # Bottom: Filtered absolute acceleration
#     # axs[1].plot(df['Absolute_filtered'], label='Filtered Absolute Acceleration')
#     # axs[1].set_title('Filtered Absolute Acceleration (MA Filter)')
#     # axs[1].set_xlabel('Time [s x 10^-2]')
#     # axs[1].set_ylabel('Acceleration [m/s²]')
#     # axs[1].legend()
#     #
#     # plt.tight_layout()
#     # plt.show()
#
#     return df
#
#
# def compare_accel_walk_jump(df_walk, df_jump, activity_label):
#     fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
#     ax.plot(df_walk['Absolute_filtered'], label='Walk Absolute Acceleration')
#     ax.plot(df_jump['Absolute_filtered'], label='Jump Absolute Acceleration')
#     ax.set_title(f'{activity_label} - Absolute Acceleration')
#     ax.set_xlabel('Time [s x 10^-2]')
#     ax.set_ylabel('Acceleration [m/s²]')
#     ax.legend()
#     plt.show()
#
#
# def create_hdf5_file():
#     # create hdf5 file
#     with h5py.File("data.h5", "w") as hdf:
#         raw_data_group = hdf.create_group("Raw data")
#         pre_processed_group = hdf.create_group("Pre-processed data")
#         segmented_data_group = hdf.create_group("Segmented data")
#
#         raw_data_group.create_group("Alex")
#         raw_data_group.create_group("George")
#         raw_data_group.create_group("Jayaram")
#
#         pre_processed_group.create_group("Alex")
#         pre_processed_group.create_group("George")
#         pre_processed_group.create_group("Jayaram")
#
#         train_group = segmented_data_group.create_group("Train")
#         test_group = segmented_data_group.create_group("Test")
#
#         names = ["Alex", "George", "Jayaram"]
#         types = ["Face", "FP", "Hand"]
#         type_data = ["Raw data", "Pre-processed data"]
#
#         for i in range(0, len(names)):
#             for k in range(0, len(types)):
#                 group_raw = hdf[f'Raw data/{names[i]}']
#                 group_process = hdf[f'Pre-processed data/{names[i]}']
#                 name = f'{names[i]}.{types[k]}'
#
#                 raw_walk_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Walking.csv')
#                 raw_jump_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Jumping.csv')
#                 process_walk_data = preprocess_data(raw_walk_data, f'{names[i]} {types[k]} Walking', 50)
#                 process_jump_data = preprocess_data(raw_jump_data, f'{names[i]} {types[k]} Jumping', 15)
#
#                 compare_accel_walk_jump(process_walk_data, process_jump_data, f'{names[i]} {types[k]}')
#
#                 group_raw.create_dataset(f'{name}.Walking', data=raw_walk_data.to_numpy())
#                 group_raw.create_dataset(f'{name}.Jumping', data=raw_jump_data.to_numpy())
#
#                 group_process.create_dataset(f'{name}.Walking', data=process_walk_data.to_numpy())
#                 group_process.create_dataset(f'{name}.Jumping', data=process_jump_data.to_numpy())
#
#
# if __name__ == '__main__':
#     create_hdf5_file()
#
#
#         # base model off of total acceleration due to differences in holding orientation of holding phone
#
#


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


# Function to preprocess data
def preprocess_data(df, activity_label, window):
    # Interpolate missing values
    df.interpolate(method='linear', inplace=True)

    window_size = window  # Window size for MA filter

    # Apply MA filter to linear accelerations
    df['x_filtered'] = df['Linear Acceleration x (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['y_filtered'] = df['Linear Acceleration y (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['z_filtered'] = df['Linear Acceleration z (m/s^2)'].rolling(window=window_size, center=True).mean()

    # Apply MA filter to existing absolute acceleration column
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
            window.max(),
            window.min(),
            window.mean(),
            window.median(),
            window.std(),
            window.max() - window.min(),
            skew(window),
            kurtosis(window),
            np.sum(np.abs(window)),
            np.count_nonzero(np.diff(np.sign(window)))
        ]

        features.append(f)

    return np.array(features)


def normalize_features(features, scaler):
    return scaler.fit_transform(features)


def create_hdf5_file():
    # create hdf5 file
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

                # Feature Extraction & Normalization
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

                # print(f"Walk features shape: {features_walk.shape}")
                # print(f"Jump features shape: {features_jump.shape}")
                # print(f"Combined features shape: {features_combined.shape}")
                # print(f"Normalized features shape: {features_normalized.shape}")
                #

if __name__ == '__main__':
    create_hdf5_file()


