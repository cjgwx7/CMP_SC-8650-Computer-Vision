import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

### Initialize argument parser
parser = argparse.ArgumentParser(description="Perform Grayscale Image Thresholding based off Visual Histogram Appraisal")

### Create required argument group
requiredNamed = parser.add_argument_group("Required named arguments")

### Add arguments
requiredNamed.add_argument("-i", "--image", help="Path to input image", required=True)
parser.add_argument("-o", "--output", help="Optional path for output image")

### Parse arguments
args = parser.parse_args()

### Save arguments
image_path = args.image
output_path = args.output

### Read in image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    sys.exit(print("Could not read image from supplied path:", image_path))

### Calculate histogram
hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])

### Display Image and Histogram
# Image
plt.figure()
plt.axis("off")
plt.title("Input Image (Grayscale)")
plt.imshow(img, cmap="gray")

# Histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency of Observed Pixels")
plt.plot(hist)
plt.xlim([0, 256])

plt.show()

### Image Thresholding

threshold = input("Input image binarization threshold: ")
threshold = np.uint8(threshold)

original_image = img.copy()
img[img < threshold] = 0
img[img > 0] = 255

### Display Results

plt.figure()
plt.title("Binarized Grayscale Image")
plt.axis("off")
plt.imshow(img, cmap="gray")

plt.figure()
plt.title("Original Grayscale Image")
plt.axis("off")
plt.imshow(original_image, cmap="gray")

plt.show()

### Optionally save results
if output_path is not None:
    cv2.imwrite(output_path, img)