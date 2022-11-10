# import necessary packages
import cv2
import cvzone as cvzone
import numpy as np
import mediapipe as mp
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.layers import Dense
from keras.models import load_model
import time
import cvzone
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import gtts
from playsound import playsound
import threading

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
emojiID = None
count = 0
reactionName = None
previousClassName = None
f.close()
# print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:

    # Read each frame from the webcam
    _, frame = cap.read()
    _, frame2 = cap.read()
    x, y, c = frame.shape

    happy_logo = cv2.imread(
        "/Users/vivekgunasekaran/Desktop/Triathlon/customer-hand-gesture-detection/images/happy.png",
        cv2.IMREAD_UNCHANGED)
    happy_logo = cv2.resize(happy_logo, (0, 0), None, 0.9, 0.9)
    x1, y1, c1 = happy_logo.shape

    sad_logo = cv2.imread("/Users/vivekgunasekaran/Desktop/Triathlon/customer-hand-gesture-detection/images/sad.png",
                          cv2.IMREAD_UNCHANGED)
    sad_logo = cv2.resize(sad_logo, (0, 0), None, 0.9, 0.9)
    x2, y2, c2 = sad_logo.shape

    awesome_logo = cv2.imread(
        "/Users/vivekgunasekaran/Desktop/Triathlon/customer-hand-gesture-detection/images/awesome.png",
        cv2.IMREAD_UNCHANGED)
    awesome_logo = cv2.resize(awesome_logo, (0, 0), None, 0.9, 0.9)
    x3, y3, c3 = awesome_logo.shape

    info_logo = cv2.imread("/Users/vivekgunasekaran/Desktop/Triathlon/customer-hand-gesture-detection/images/info.png",
                           cv2.IMREAD_UNCHANGED)
    info_logo = cv2.resize(info_logo, (0, 0), None, 1.2, 1.2)
    x4, y4, c4 = info_logo.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    frame2 = cv2.flip(frame2, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])

            classID = np.argmax(prediction)

            emojiID = classID
            className = classNames[classID]

    # show the prediction on the frame
    reactionName = className
    cv2.putText(frame, className, (650, 100), cv2.FONT_HERSHEY_TRIPLEX,
                4, (0, 255, 255), 4, cv2.FILLED)

    if emojiID == 2:
        imgResult = cvzone.overlayPNG(frame, happy_logo, [0, x1 - x])
        cv2.imshow("Output", imgResult)
        if className != previousClassName:
            customerReport(className)
            previousClassName = className
            callSpeech(className)

    elif emojiID == 3:
        imgResult = cvzone.overlayPNG(frame, sad_logo, [0, x2 - x])
        cv2.imshow("Output", imgResult)
        if className != previousClassName:
            customerReport(className)
            previousClassName = className
            callSpeech(className)

    elif emojiID == 0:
        imgResult = cvzone.overlayPNG(frame, awesome_logo, [0, x3 - x])
        cv2.imshow("Output", imgResult)
        if className != previousClassName:
            customerReport(className)
            previousClassName = className
            callSpeech(className)

    else:
        imgResult = cvzone.overlayPNG(frame2, info_logo, [1150, 0])
        cv2.imshow("Output", imgResult)

    #This Method use to convert customer reaction to speech
    def callSpeech(classNameSpeech):
        if classNameSpeech != '':
            tts = gtts.gTTS(classNameSpeech)
            tts.save("hello.mp3")
            threading.Thread(target=playsound, args=('hello.mp3',), daemon=True).start()

    def customerReport(reactionName):
        df = pd.read_csv('Customer_Reactions.csv')
        if reactionName == 'GOOD':
            column_value = df.iloc[1, 1]
            new_value = column_value + 1
            df.at[1, 'Count'] = new_value
            df.to_csv('Customer_Reactions.csv', index=False)
        elif reactionName == 'DISAPPOINTED':
            column_value = df.iloc[2, 1]
            new_value = column_value + 1
            df.at[2, 'Count'] = new_value
            df.to_csv('Customer_Reactions.csv', index=False)
        elif reactionName == 'AWESOME':
            column_value = df.iloc[0, 1]
            new_value = column_value + 1
            df.at[0, 'Count'] = new_value
            df.to_csv('Customer_Reactions.csv', index=False)


    k = cv2.waitKey(1)

    # Press Escape key to close the application
    if k % 256 == 27:
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()

# view the report
df = pd.read_csv('Customer_Reactions.csv')
# plt.bar(courses, values, color ='maroon',width = 0.4)
plt.rcParams.update({'font.size': 7})
df.plot(x='Reaction', y='Count', kind='bar', color='maroon', width=0.2, figsize=(10, 7))
plt.title("Customer Ratings", fontsize=18)
plt.xlabel("Reactions", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()
