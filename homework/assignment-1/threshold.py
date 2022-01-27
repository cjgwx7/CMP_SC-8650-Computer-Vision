
import argparse
import cv2
import numpy as np
import sys
import os

### Initialize argument parser
parser = argparse.ArgumentParser(description="Perform Grayscale Image Thresholding")

### Create required argument group
requiredNamed = parser.add_argument_group("Required named arguments")

### Add arguments
requiredNamed.add_argument("-i", "--image", help="Path to input image", required=True)
requiredNamed.add_argument("-t", "--threshold", help="Pixel threshold value", type=int, required=True)
parser.add_argument("-o", "--output", help="Optional path for output image")

### Parse arguments
args = parser.parse_args()

### Save arguments
image_path = args.image
threshold = np.uint8(args.threshold)
output_path = args.output

### Read in image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    sys.exit(print("Could not read image from supplied path:", image_path))

### Perform image thresholding
img[img <= threshold] = 0
img[img > 0] = 255

### Display image
cv2.imshow("Display window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if output_path is not None:
    cv2.imwrite(output_path, img)