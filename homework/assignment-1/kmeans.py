import cv2
import numpy as np
import matplotlib.pyplot as plt

### Read in image as grayscale or color based on flag
image = cv2.imread("C:/Users/cgroh/OneDrive/Desktop/PhD/courses-program/SP22/CMP_SC-8650-Computer-Vision/homework/assignment-1/images/building.jpg", 0)
dim = (224, 224)
image = cv2.resize(image, dim)
row = image.shape[0]
col = image.shape[1]
channel = 1
image = image.reshape(row*col, channel)
print(image)

### KMeans algorithm
def pairwise_distance(x, y):
    SS_x = np.sum(np.square(x), axis = 1)
    SS_y = np.sum(np.square(y), axis = 1)
    product = np.dot(SS_x, SS_y.T)
    distances = np.uint8(np.sqrt(abs(SS_x[:, np.newaxis] + SS_y-2*product)))
    return distances

dist = pairwise_distance(x = image, y = image)
print(np.max(dist))