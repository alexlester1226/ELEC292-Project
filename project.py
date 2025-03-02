# Raw Data Project Python File
# Authors:
# Alex Lester - 20411355
# George Choueiry - 20411131
# Jayaram Jeyakanthan 20410793

# import statements
import pandas as pd
import numpy as np
import h5py


data = pd.read_csv('./Raw Data/Alex/Alex.FP.Walking.csv')
print(data)


# create hdf5 file
with h5py.File("data.h5", "w") as hdf:

    # Create the top-level groups
    raw_data_group = hdf.create_group("Raw data")
    pre_processed_group = hdf.create_group("Pre-processed data")
    segmented_data_group = hdf.create_group("Segmented data")

    # Create subgroups under Raw data
    raw_data_group.create_group("Alex Lester")
    raw_data_group.create_group("George Choueiry")
    raw_data_group.create_group("Jayaram Jeyakanthan")

    # Create subgroups under Pre-processed data
    pre_processed_group.create_group("Alex Lester")
    pre_processed_group.create_group("George Choueiry")
    pre_processed_group.create_group("Jayaram Jeyakanthan")

    # Create subgroups under Segmented data
    train_group = segmented_data_group.create_group("Train")
    test_group = segmented_data_group.create_group("Test")




    # Example of writing a dataset
    train_group.create_dataset("sample_dataset", data=[1, 2, 3, 4, 5])

print("HDF5 file created successfully with the specified structure.")



