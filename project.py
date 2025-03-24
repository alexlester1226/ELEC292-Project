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


# create hdf5 file
with h5py.File("data.h5", "w") as hdf:

    # Create the top-level groups
    raw_data_group = hdf.create_group("Raw data")
    pre_processed_group = hdf.create_group("Pre-processed data")
    segmented_data_group = hdf.create_group("Segmented data")

    # Create subgroups under Raw data
    raw_data_group.create_group("Alex")
    raw_data_group.create_group("George")
    raw_data_group.create_group("Jayaram")

    # Create subgroups under Pre-processed data
    pre_processed_group.create_group("Alex")
    pre_processed_group.create_group("George")
    pre_processed_group.create_group("Jayaram")

    # Create subgroups under Segmented data
    train_group = segmented_data_group.create_group("Train")
    test_group = segmented_data_group.create_group("Test")

    # string labels used in CSV file labeling
    names = ["Alex", "George", "Jayaram"]
    types = ["Face", "FP", "Hand"]

    # for loop add Raw CSV Data to groups
    for i in range(0, len(names)):
        group = hdf[f'Raw data/{names[i]}']  # locate specific group

        for k in range(0, len(types)):
            name = f'{names[i]}.{types[k]}'

            # get walking and jumping data and convert it using ".to_numpy()"
            walk_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Walking.csv').to_numpy()
            jump_data = pd.read_csv(f'Raw Data/{names[i]}/{name}.Jumping.csv').to_numpy()

            # creating datasets with the converted csv data
            group.create_dataset(f'{name}.Walking', data=walk_data)
            group.create_dataset(f'{name}.Jumping', data=jump_data)




# Line graph for x, y, z acceleration
df = pd.read_csv('Raw Data/Alex/Alex.FP.Walking.csv')

fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

ax.plot(df['Linear Acceleration x (m/s^2)'], label='X-axis')
ax.plot(df['Linear Acceleration y (m/s^2)'], label='Y-axis')
ax.plot(df['Linear Acceleration z (m/s^2)'], label='Z-axis')

ax.set_title('Accelerometer Data - Alex FP Walking')
ax.set_xlabel('Sample Number (Time Index)')
ax.set_ylabel('Acceleration (m/s²)')
ax.legend()
plt.show()

df = pd.read_csv('Raw Data/Alex/Alex.FP.Jumping.csv')

fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

ax.plot(df['Linear Acceleration x (m/s^2)'], label='X-axis')
ax.plot(df['Linear Acceleration y (m/s^2)'], label='Y-axis')
ax.plot(df['Linear Acceleration z (m/s^2)'], label='Z-axis')

ax.set_title('Accelerometer Data - Alex FP Jumping')
ax.set_xlabel('Sample Number (Time Index)')
ax.set_ylabel('Acceleration (m/s²)')
ax.legend()
plt.show()


# Line graph for x, y, z acceleration
df = pd.read_csv('Raw Data/Alex/Alex.FP.Walking.csv')

fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

ax.plot(df['Absolute acceleration (m/s^2)'], label='X-axis')


ax.set_title('Accelerometer Data - Alex FP Walking')
ax.set_xlabel('Sample Number (Time Index)')
ax.set_ylabel('Acceleration (m/s²)')
ax.legend()
plt.show()

# Line graph for x, y, z acceleration
df = pd.read_csv('Raw Data/Alex/Alex.FP.Jumping.csv')

fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

ax.plot(df['Absolute acceleration (m/s^2)'], label='X-axis')


ax.set_title('Accelerometer Data - Alex FP Jumping')
ax.set_xlabel('Sample Number (Time Index)')
ax.set_ylabel('Acceleration (m/s²)')
ax.legend()
plt.show()
