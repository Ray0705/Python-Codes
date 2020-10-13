import os
from PIL import Image
import numpy as np
import cv2
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,"images")

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
# pip install opencv-contrib-python --user
# use above installation if it shows any error while recognizer
# AttributeError: module 'cv2.cv2' has no attribute 'face'
# If above error occurs


current_id = 0
label_ids = {}
x_train = []
y_labels = []

# to check whether the we are in correct directory or not
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").capitalize()
            # print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            pil_image = Image.open(path).convert("L") # convert "L" converts the image into grayscale
            size = (550,500)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("labels.pkl","wb") as f:
    pickle.dump(label_ids,f)


recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")


