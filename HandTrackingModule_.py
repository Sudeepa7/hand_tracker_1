import cv2
import mediapipe as mp
import time

class handDetector():

    def __init__(self,mode=False,maxHands=2,modecompl=1,detectionCon=0.5,trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.modecompl = modecompl
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        #Creates a Hands object from the Mediapipe library for hand tracking
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.modecompl,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):

        # Converts the BGR image (OpenCV default) to RGB format, which is required by Mediapipe.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Processes the RGB image with the Mediapipe Hands model to detect hand landmarks.
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draws the landmarks on the frame using the mpDraw module from Mediapipe.
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,handNo=0,draw=True):

        #landmark list that we are going to return
        lmList =[]

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                #if id == 4:
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector()
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


if __name__ =="__main__":
    main()

