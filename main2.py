import cv2
import numpy as np

# capturing the video from camera
cap = cv2.VideoCapture(0)

# converting the names from file to a list
className = []
classFile = 'coco.names'
with open (classFile,'rt') as file:
    className = file.read().rstrip()