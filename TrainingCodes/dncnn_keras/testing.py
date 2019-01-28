import os, time, datetime
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread, imsave

#'C:/Users/cvlab/Documents/GitHub/DnCNN/TrainingCodes/dncnn_keras/results/Set68/test001_dncnn.png'
#'C:/Users/cvlab/Documents/GitHub/DnCNN/TrainingCodes/dncnn_keras/data/Test/Set68/test001.png'
x = np.array(imread('C:/Users/cvlab/Documents/GitHub/DnCNN/TrainingCodes/dncnn_keras/data/Test/Set68/test003.png', dtype=np.float32)) / 255.0
print(x)


import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))
plt.imshow(x,interpolation='nearest',cmap='gray')
plt.show()

'''
x = cv2.imread('C:/Users/cvlab/Documents/GitHub/DnCNN/TrainingCodes/dncnn_keras/data/Test/Set68/test003.png', cv2.CV_8UC1)
denoised_image = cv2.imread('C:/Users/cvlab/Documents/GitHub/DnCNN/TrainingCodes/dncnn_keras/results/Set68/test003_dncnn.png', cv2.CV_8UC1)
noise = np.random.normal(0, 25/255.0, x.shape)
cv2.imshow('original', x)
cv2.waitKey(1000)
noise_image = x + noise
noise_image = noise_image.astype(np.float32)
#noise_image = np.clip(noise_image, 0, 255)
print(noise_image,'\n', np.histogram(noise_image))
print('---------------')
cv2.imshow('noise image', noise_image)
cv2.waitKey(1000)

cv2.imshow('denoised image', denoised_image)
cv2.waitKey(1000)

residual = denoised_image - noise_image
print(residual, np.histogram(residual))
#residual = residual.astype(np.uint8)
#print(np.histogram(residual))
#residual = cv2.equalizeHist(residual)
print(np.histogram(residual))
cv2.imshow('noise-denoise', residual)
cv2.waitKey(8000)
cv2.destroyAllWindows()
'''
