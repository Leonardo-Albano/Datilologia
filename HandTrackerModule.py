import os
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
                    
    
    def findPosition(self, img, handNumber=0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(hand.landmark):
                        h, w, c = img.shape
                        cx , cy = int(lm.x*w), int(lm.y*h)
                        lmList.append([id, cx, cy])
                        if draw:
                            print(id, cx, cy) 
                        cv2.circle(img, (cx,cy), 5, (255, 0, 255), cv2.FILLED)
    
        return lmList


# def main():

#     pTime = 0
#     oTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = handDetector()


#     while True:
#         sucess, img = cap.read()
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
#         if len(lmList) != 0: 
#             print(lmList[4])

#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime
        
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)

# if __name__ == "__main__":
#     main()