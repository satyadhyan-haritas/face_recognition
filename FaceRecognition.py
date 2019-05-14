#!/usr/bin/env python
# coding: utf-8

# ### Images are expected in the directory  "<current dir>/images/train/"
# #### Eg: if current directory is Desktop then images should be in -
# #### Desktop/images/train

# In[3]:


import os
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import keras
from PIL import Image
import face_recognition
import cv2
import glob
import traceback


# In[4]:


face_encodings = []
filenames = []

def loadImages(image_dir):
    cwd = os.getcwd()
    if(not os.path.exists(image_dir)):
        #print("present wd:",cwd)
        print("Skipped ",image_dir ," as doesn't exist")
        return
    
    os.chdir(image_dir)
    #print("prev wd:",cwd , "\n  curr wd:",os.getcwd())
    
    files = glob.glob("*.jp*")
    percent = 1
    random_files = files
    prepend_dir = False
    
    if len(files) > 100:
        prepend_dir = True
        random_files = np.random.choice(files, 10)
        #percent = 0.01
        #random_files = np.random.choice(files, int(len(files) * percent))
        
    #print("Actual files: in ",image_dir, "=", len(files)," random_files: ",len(random_files))
    for filename in random_files:
        try:
            with open(filename,'r') as f:
                name = filename.split(".")[0]
                if(prepend_dir):
                    name = os.path.basename(os.getcwd()) + "-" + name
                face = face_recognition.load_image_file(filename)
                filter_face_encoding = face_recognition.face_encodings(face)    
                if len(filter_face_encoding) == 0:
                    continue
                face_encodings.append(filter_face_encoding[0])
                filenames.append(name)
        except Exception as ex:
            tb = traceback.format_exc()
            os.chdir(cwd)
            print(os.getcwd())
            print("Error loading images\n",tb)
            
    os.chdir(cwd)
    #print("-----  curr wd:",os.getcwd())


# # Give the image directories individually below

# In[16]:


loadImages("Images/train/AkshayKumar/")
loadImages("Images/train/Amitabh/")


# In[18]:


known_face_encodings = face_encodings
known_face_names = filenames
len(known_face_names)


# In[19]:


# Initialize some variables
test_face_locations = []
test_face_encodings = []
test_face_names = []
process_this_frame = True


# In[20]:


video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.40)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name.split("-")[0])

    process_this_frame = not process_this_frame

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
    #for (top, right, bottom, left) in faces, name in zip(face_locations, face_names)
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Seethos Face Recognition', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()






