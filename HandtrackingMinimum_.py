import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# Creates a Hands object from the Mediapipe library for hand tracking
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    # Converts the BGR image (OpenCV default) to RGB format, which is required by Mediapipe.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Processes the RGB image with the Mediapipe Hands model to detect hand landmarks.
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            # Draws the landmarks on the frame using the mpDraw module from Mediapipe.
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    # how many frames were processed in one second.
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    # Waits for a key press for 1 millisecond. If a key is pressed,
    # it proceeds to the next iteration of the loop. This allows for smooth video playback.
    cv2.waitKey(1)
