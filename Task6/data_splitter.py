import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

def create_folders(base_dir, categories):
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)

def copy_file(src, dest):
    with open(src, 'rb') as fs:
        contents = fs.read()
    with open(dest, 'wb') as fd:
        fd.write(contents)

def split_data_without_shutil(source, training, testing, split_size):
    for category in os.listdir(source):
        category_path = os.path.join(source, category)
        if not os.path.isdir(category_path):
            continue
        
        images = os.listdir(category_path)
        
        train_images, test_images = train_test_split(images, test_size=(1 - split_size), random_state=42)
        
        for image in train_images:
            src_path = os.path.join(category_path, image)
            dest_path = os.path.join(training, category, image)
            copy_file(src_path, dest_path)
        
        for image in test_images:
            src_path = os.path.join(category_path, image)
            dest_path = os.path.join(testing, category, image)
            copy_file(src_path, dest_path)

def main():
    # Define paths
    data_dir = './data'
    train_dir = './my_data/edge_detected/training'
    test_dir = './my_data/edge_detected/testing'

    categories = ['cups', 'plates', 'bowls']
    
    # Create train and test directories
    create_folders(train_dir, categories)
    create_folders(test_dir, categories)
    
    # Split the data
    split_size = 0.8 # 80% training, 20% testing
    split_data_without_shutil(data_dir, train_dir, test_dir, split_size)

if __name__ == "__main__":
    main()