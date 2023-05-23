import HandTrackerModule as htm
import time
import cv2
import pickle
import numpy as np

pTime = 0
oTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)
modelDict = pickle.load(open('./model.p', 'rb'))
model = modelDict['model']
letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L', 9: 'M', 10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 19: 'W'}

while True:
    sucess, img = cap.read()
    img = detector.findHands(img, False)
    landmarks = detector.findPosition(img, draw=False)
    dataAUX = []

    if landmarks: 
        for points in landmarks:
                id = points[0]
                x = points[1]
                y = points[2]
                dataAUX.append(x)
                dataAUX.append(y)
        prediction = model.predict([np.asarray(dataAUX)])
        predictedCaracter = letters[int(prediction[0])]
        cv2.putText(img, predictedCaracter, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # print(predictedCaracter)

    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()