import cv2 as cv
import numpy as np

#capturing the video by the default camera
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#open the file using read() and convert the names in the file into list
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as file:
    classNames = file.read().rsplit('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Deep Neural Network (DNN) module, which comes with a nice pre-trained face detection convolutional CNN
net = cv.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    _,img = cap.read()

    # conf - accuracy of detected object
    # bbox - bounding box
    # classIds - index of the names
    classIds, confs, bbox = net.detect(img,confThreshold=0.6)
    print(classIds,bbox,confs)

    if len(classIds) != 0:
        for classId , confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv.rectangle(img, box, color=(0,255,0),thickness=3)
            cv.putText(img,classNames[classId -1],(box[0]+10,box[1]+30),
                       cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv.imshow('Output-frame', img)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
