import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 22
eraserThickness = 160
########################

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)[0]

    if len(lmList) != 0 and len(lmList) != 2:
        # print(lmList)
        # print("got it : " , lmList[8])
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2] and fingers[3] == False:
            xp, yp = 0, 0
            print("Selection Mode")
            print("         ", x1, y1)
            # 450, 620, 750, 950 pixels
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (128, 128, 0)
                elif 500 < x1 < 620:
                    header = overlayList[1]
                    drawColor = (0, 255, 255)
                elif 670 < x1 < 750:
                    header = overlayList[2]
                    drawColor = (255, 0, 255)
                elif 780 < x1 < 870:
                    header = overlayList[3]
                    drawColor = (0, 255, 0)
                elif 900 < x1 < 1200:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

        else:
            xp, yp = 0, 0

        # Clear Canvas when all fingers are up
        # if all (x >= 1 for x in fingers):
        # if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
        #     imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)                    #to draw on canvas
    # cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
