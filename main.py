import cv2

img = cv2.imread('Obama.png')

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames= f.read().rstrip('\n').split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


classIds, confs , bbox = net.detect(img,confThreshold=0.5)
print(classIds,bbox )

cv2.imshow("Output",img)
cv2.waitKey(0)
