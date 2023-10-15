import os
import shutil

# Source folder containing the .jpg files
source_folder = '/home/andy/ICASSP_data/data/duke/bounding_box_all'

# Destination folder to copy the files
destination_folder = '/home/andy/ICASSP_data/data/data_folder/duke/all_folder'
dataset = '_duke'
# Iterate through the .jpg files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.jpg'):
        # Extract the first four characters from the file name
        prefix = filename[:4]+dataset

        # Create the destination folder if it doesn't exist
        folder_path = os.path.join(destination_folder, prefix)
        os.makedirs(folder_path, exist_ok=True)

        # Copy the file to the destination folder
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(folder_path, filename)
        shutil.copyfile(source_path, destination_path)