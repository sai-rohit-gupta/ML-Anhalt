"""
Title: Task Processing Kitchenware Images
Author: Subash Rajanayagam
Date: 04/2022
Description: 
# Necessary libraries via import
# URLS for OpenCV2: 
# Resoruce: https://www.andreasjakl.com/basics-of-ar-anchors-keypoints-feature-detection/pip-install-opencv-python/
# make sure you add python3 instead of python before your pip command if you have multiple
# python enviornments on your machine
"""



import cv2
import numpy as np
from matplotlib import pyplot as plt

# Loop to read and write automatic edge-detected images 
# Follow opencv tutorials at: 
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
# Make sure you use adjust your directory path correctly
classes = ['cups', 'Bowls', 'plates']

for num, clas in enumerate(classes):
    for x in range(0,25,10):
        img = cv2.imread(f'./data/{clas}/{num+1}.{x+1}.jpg',0)
        print(img.shape)
        edges = cv2.Canny(img,100,200)
        img = cv2.imwrite(f'./data/my_data/edge_detected/test/{clas}/{num+1}.{x+1}.jpg',edges)

# Visualizing grayscaled and edged images
        img = cv2.imread(f'./data/{clas}/{num+1}.{x+1}.jpg', 0)
        edges = cv2.Canny(img,100,200)
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()

# Applying various threshing of images
# Check the example at:
#  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
# More image processing can be found at: 
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])

        plt.show()

        # global thresholding
        ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # plot all the images and their histograms
        images = [img, 0, th1,
                img, 0, th2,
                blur, 0, th3]
        titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
                'Original Noisy Image','Histogram',"Otsu's Thresholding",
                'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

        for i in range(3):
            plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
            plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
            plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
            plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        plt.show()