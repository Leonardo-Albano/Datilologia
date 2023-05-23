import os
import mediapipe as mp
import cv2
import HandTrackerModule as htm
import matplotlib.pyplot as plt
import pickle

DATA_DIR = './data'
detector = htm.handDetector(True, 1, 0.3)

count = 0
data = []
labels = []
for dir in os.listdir(DATA_DIR):
    for imgPath in os.listdir(os.path.join(DATA_DIR, dir)):
        dataAUX = []
        img = cv2.imread(os.path.join(DATA_DIR, dir, imgPath))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgRGB = detector.findHands(imgRGB, draw=False)
        landmarks = detector.findPosition(imgRGB, draw=False)

        if(landmarks):
            for points in landmarks:
                id = points[0]
                x = points[1]
                y = points[2]
                dataAUX.append(x)
                dataAUX.append(y)
            data.append(dataAUX)
            labels.append(dir)
            
    print(count)
    count += 1

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()



        
    