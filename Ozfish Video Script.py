# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:53:10 2022

@author: Denzel
"""

import cv2
import pickle
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import tensorflow_addons # For focal loss function used in model

model = load_model('ozfish_cnn')
labels = pickle.load(open('ozfish_labels.pkl', 'rb'))
queue = deque(maxlen = 90)
clip = cv2.VideoCapture('video/BRUVS_12.mp4')
writer = None
(W, H) = (None, None)

while True:
    exist, frame = clip.read()
    if not exist:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    output = frame.copy() # Save original frame for output later
    # Preprocess the frame to match properties of images used in training
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB format
    frame = cv2.resize(frame, (128, 128)).astype('float32') # Resize frame to same as trained images: (128, 128)
    frame = frame/255.0 # Rescale so pixel values are between 0 and 1
    frame = np.expand_dims(frame, axis = 0) # Expand dims to match tensor shape
    # Use model to make predictions on frame
    prediction = model.predict(frame)
    queue.append(prediction) # Push prediction onto queue
    # Calculate rolling average of current predictions
    avg = np.array(queue).mean(axis = 0) # Average probability of each class based on current history
    index = np.argmax(avg) # Get index of class with highest average probability
    species = labels[index]
    if cv2.waitKey(1) == ord('q'):
        break
    # Format the output image
    text = "Species: {}".format(species)
    org = (0,55)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    cv2.putText(output, text, org, font, fontScale, color, thickness) # Add text to image
    # Write the output video to disk
    if writer == None:
        fps = 30
        (H, W) = output.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # .mp4 format
        writer = cv2.VideoWriter('labelled/BRUVS_Labelled_4.mp4', fourcc, fps, (W, H))
    writer.write(output)
    
writer.release()
clip.release()