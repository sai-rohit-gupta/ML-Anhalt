"""
Title: Task Load Image and Process
Author: Subash Rajanayagam
Date: 04/2022
Description: 
# Necessary libraries via import
# make sure to install python 3.9 at: https://www.python.org/downloads/
# Install and setup VS Code at: https://code.visualstudio.com/docs/python/python-tutorial
# Install tensorflow and keras at: https://www.tensorflow.org/install
# Install  numpy and matplotlib at: https://solarianprogrammer.com/2017/02/25/install-numpy-scipy-matplotlib-python-3-windows/
"""


import tensorflow as tf             
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Desired image resolution
img_height = 128
img_width = 128

# Location of train and test images
data_path_train = './my_data/edge_detected/training'
data_path_test = './my_data/edge_detected/testing'

# Prepare training data from image to tf.data.Dataset
# Check the tutorial: https://www.tensorflow.org/tutorials/load_data/images
# For seed,guarantee the same set of randomness [e.g. initializing weights of ANN, if not set, very different results can arrise]
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_train,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = False,
    seed=123,
)
print("below is train data.")
print(ds_train)

# Prepare test data from image to tf.data.Dataset
# For seed,guarantee the same set of randomness [e.g. initializing weights of ANN, if not set, very different results can arrise] 
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_test,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = False,
    seed=123,                                 
)
print("below is test data.")
print(ds_validation)

# Retriving class names from the tf.data.Dataset of training data
class_names = ds_train.class_names
print(class_names)

# Retriving and displaying training images from tf.data.Dataset of training data
plt.figure(figsize=(10, 10))
for images, labels in ds_train:
    for i in range(20):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

# Retriving class names from the tf.data.Dataset of validation data
class_names = ds_validation.class_names
print(class_names)

# Retriving and displaying training images from tf.data.Dataset of validation data
plt.figure(figsize=(10, 10))
for images, labels in ds_validation.take(-1):
    for i in range(30):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()  