import os

# Path to the directory containing the files
folder_path = './data/cups'

# Function to rename jpeg files to jpg
def rename_files_in_folder(folder):
    for filename in os.listdir(folder):
        # Check if the file has a .jpeg extension
        if filename.lower().endswith('.jpeg'):
            # Determine the new filename with a .jpg extension
            new_filename = filename[:-5] + '.jpg'
            
            # Full paths to the old and new files
            old_filepath = os.path.join(folder, filename)
            new_filepath = os.path.join(folder, new_filename)
            
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f'Renamed: {filename} -> {new_filename}')

# Rename jpeg to jpg in the specified folder
rename_files_in_folder(folder_path)