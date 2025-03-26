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


# ===============================
# STEP 1–3: Data Storage (same as before)
# ===============================

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

    for i in range(len(names)):
        group = hdf[f'Raw data/{names[i]}']
        for k in range(len(types)):
            name = f'{names[i]}.{types[k]}'
            walk_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Walking.csv').to_numpy()
            jump_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Jumping.csv').to_numpy()
            group.create_dataset(f'{name}.Walking', data=walk_data)
            group.create_dataset(f'{name}.Jumping', data=jump_data)


# ===============================
# STEP 4: PRE-PROCESSING
# ===============================

# Function to preprocess data
def preprocess_data(filepath, activity_label):
    # Load the raw data
    df = pd.read_csv(filepath)

    # Rename columns for convenience (adjust based on actual headers)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # ===== 1. Handle Missing Data =====
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

    plt.xlabel('Sample Index')
    plt.tight_layout()
    plt.show()

    return df


# ===============================
# EXAMPLE CALLS: Preprocess and Plot
# ===============================

# Alex - FP - Walking
walk_path = 'Raw Data/Alex/Alex.FP.Walking.csv'
df_walk = preprocess_data(walk_path, 'Alex FP Walking')

# Alex - FP - Jumping
jump_path = 'Raw Data/Alex/Alex.FP.Jumping.csv'
df_jump = preprocess_data(jump_path, 'Alex FP Jumping')
