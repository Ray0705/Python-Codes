import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}
with open("labels.pkl",'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        # img_item = '10.png'
        roi_color = frame[y:y+h,x:x+w]
        id_,conf = recognizer.predict(roi_gray)
        if conf>=70: # and conf<=85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,0,0)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,0.75,color,stroke,cv2.LINE_AA)

        # cv2.imwrite(img_item,roi_color)
        color = (0,255,0)
        stroke = 2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()