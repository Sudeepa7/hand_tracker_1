import cv2
import mediapipe as mp
import time
import HandTrackingModule_ as htm

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[4])

    cTime = time.time()
    # how many frames were processed in one second.
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    # Waits for a key press for 1 millisecond. If a key is pressed,
    # it proceeds to the next iteration of the loop. This allows for smooth video playback.
    cv2.waitKey(1)
