# All at one place
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image

st.header("YOLO(v3) Object Detection")

uploaded_file = st.file_uploader("Choose a image file",type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img=Image.open(uploaded_file)
    st.image(img)

    modelConf="yolov3-tiny.cfg"
    modelWeights="yolov3-tiny.weights"
    classesFile="coco.names"

    def post_process(frame,outs,img,classes):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        boxes=[]
        confidences=[]
        classIDs=[]
        for out in outs: # calling each object boxes
            for detection in out: # calling each box
                score=detection[5:] # probability of 80 classes
                class_id=np.argmax(score) # max probability id
                confidence=score[class_id] # getting the confidance
                if confidence>0.7:         # if confidance >70% consider as that is valid bounding box
                    centerX = int(detection[0] * frameWidth)  # before we pass the object we divided with frame width
                    # these are the normalized values so multiply again
                    centerY = int(detection[1] * frameHeight)
                    width = int(detection[2]* frameWidth)
                    height = int(detection[3]*frameHeight )
                    left = int(centerX - width/2)
                    top = int(centerY - height/2)
                    classIDs.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,0,0) # RGB
        for i in indexes:
            x,y,w,h=boxes[i]
            label=str(classes[classIDs[i]])
            confi=str(round(confidences[i],2))
            cv2.rectangle(img,(x,y),(x+w,y+h),color,5,i)   # (x,y): left,top     (x+w,y+h): right,bottom
            cv2.putText(img,label +" "+confi,(x,y),font,2,(255,255,255),3)

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert NumPy array to PIL image
        pil_image = Image.fromarray(image_rgb)
        st.image(pil_image)

    def yolo_live(modelConf,modelWeights,classesFile,img):
        net = cv2.dnn.readNetFromDarknet(modelConf,modelWeights)
        with open(classesFile,'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        #cap = cv2.VideoCapture(0)

     
        frame = np.array(img)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        inpWidth = 416
        inpHeight = 416
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False) # pass the image
        net.setInput(blob)
        yolo_layers=net.getUnconnectedOutLayersNames()
        outs = net.forward(yolo_layers)
        post_process(frame,outs,img,classes)

    yolo_live(modelConf,modelWeights,classesFile,img)