import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
numberOfClasses = 20
datasetSize = 100

cap = cv2.VideoCapture(1)
for j in range(numberOfClasses):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    
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
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)
        
        counter += 1
        
cap.release()
cv2.destroyAllWindows()