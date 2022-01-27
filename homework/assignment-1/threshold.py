
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
threshold = args.threshold
output_path = args.output

print(image_path, threshold, output_path)



