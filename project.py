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


# Function to preprocess data
def preprocess_data(df, activity_label):

    # Interpolate to fill missing values (NaNs)
    df.interpolate(method='linear', inplace=True)

    # ===== 2. Apply Moving Average Filter =====
    window_size = 5  # You can tweak this value for best results

    df['x_filtered'] = df['Linear Acceleration x (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['y_filtered'] = df['Linear Acceleration y (m/s^2)'].rolling(window=window_size, center=True).mean()
    df['z_filtered'] = df['Linear Acceleration z (m/s^2)'].rolling(window=window_size, center=True).mean()

    # ===== 3. Plot Before and After Filtering =====
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{activity_label} - Raw vs Filtered Accelerometer Data')


    axs[0].plot(df['Linear Acceleration x (m/s^2)'], label='Raw X')
    axs[0].plot(df['Linear Acceleration y (m/s^2)'], label='Raw Y')
    axs[0].plot(df['Linear Acceleration z (m/s^2)'], label='Raw Z')
    axs[0].set_title('Raw Acceleration')
    axs[0].legend()

    axs[1].plot(df['x_filtered'], label='Filtered X')
    axs[1].plot(df['y_filtered'], label='Filtered Y')
    axs[1].plot(df['z_filtered'], label='Filtered Z')
    axs[1].set_title('Filtered Acceleration (MA filter)')
    axs[1].legend()

    plt.xlabel('Time [ms]')
    plt.ylabel('Acceleration [m/s^2]')

    plt.tight_layout()
    plt.show()

    return df


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
    type_data = ["Raw data", "Pre-processed data"]

    for i in range(0, len(names)):
        for k in range(0, len(types)):
            group_raw = hdf[f'Raw data/{names[i]}']
            group_process = hdf[f'Pre-processed data/{names[i]}']
            name = f'{names[i]}.{types[k]}'

            raw_walk_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Walking.csv')
            raw_jump_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Jumping.csv')
            process_walk_data = preprocess_data(raw_walk_data, f'{names[i]} {types[k]} Walking')
            process_jump_data = preprocess_data(raw_jump_data, f'{names[i]} {types[k]} Jumping')

            group_raw.create_dataset(f'{name}.Walking', data=raw_walk_data.to_numpy())
            group_raw.create_dataset(f'{name}.Jumping', data=raw_jump_data.to_numpy())

            group_process.create_dataset(f'{name}.Walking', data=process_walk_data.to_numpy())
            group_process.create_dataset(f'{name}.Jumping', data=process_jump_data.to_numpy())






