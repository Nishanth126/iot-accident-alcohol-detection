import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade/haarcascade_righteye_2splits.xml')
eyes = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

lbl = ['Closed eyes', 'Open eyes']

model = load_model('CNN__model.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
flag = 0
# Initialize predictions with default values
rpred = [[0.5]]  # Initialize as 2D array
lpred = [[0.5]]  # Initialize as 2D array

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    eye = eyes.detectMultiScale(gray)
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    
    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Draw rectangles for visualization
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (150, 150, 150), 1)
        
    for (x, y, w, h) in eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (150, 150, 150), 1)
    
    # Process right eye
    r_eye_detected = False
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (100, 100))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 100, 100, 1)
        rpred = model.predict(r_eye, verbose=0)
        r_eye_detected = True
        break

    # Process left eye  
    l_eye_detected = False
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (100, 100))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 100, 100, 1)
        lpred = model.predict(l_eye, verbose=0)
        l_eye_detected = True
        break

    # Determine eye state and update score
    if r_eye_detected and l_eye_detected:
        # Assuming your model outputs probability of eye being open
        # Adjust the threshold based on your model's output
        if rpred[0][0] < 0.3 and lpred[0][0] < 0.3:  # Both eyes closed
            score += 1
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:  # At least one eye open
            score = max(0, score - 1)  # Ensure score doesn't go below 0
            cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # If eyes not detected, assume open (conservative approach)
        score = max(0, score - 1)
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Display score
    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score < 2 :
        flag = 0
    if score > 10  and flag == 0:
        try:
            from serial_test import Send
            Send("A")
            flag= 1
        except Exception as e:
            print("An error occurred:", e)
    # Trigger alarm if score exceeds threshold
    if score > 10:

        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
            
        # Visual alert - flashing rectangle
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    
    cv2.imshow('Driver drowsiness detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()