import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
numberOfClasses = 20
datasetSize = 100
letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L', 9: 'M', 10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 19: 'W'}

cap = cv2.VideoCapture(0)
for j in range(numberOfClasses):
    letter = letters[j]
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(letter))
    
    done = False
    while True:
        ret, img = cap.read()
        cv2.putText(img, 'Press Q to start! ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('image', img)
        if cv2.waitKey(25) == ord('q'):
            break
        
    counter = 0
    while(counter < datasetSize):
        ret, img = cap.read()
        cv2.imshow('image', img)
        cv2.waitKey(70)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)
        
        counter += 1
        
cap.release()
cv2.destroyAllWindows()