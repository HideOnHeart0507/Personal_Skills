import math

import cvzone
import cv2
import numpy as np
import mediapipe as mp
import random
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
# mphands = mp.solutions.hands
# hands = mphands.Hands
detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeGameClass:
    def __init__(self,pathFood):
        self.points = [] # all points
        self.lengths = [] # distance between each point
        self.currentlength = 0 # current length of the snake
        self.limitlength = 150 # total allowed length
        self.previoushead = 0,0 #previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imgFood = cv2.resize(self.imgFood, (75,75), interpolation=cv2.INTER_AREA)
        self.hfood,self.wfood, _ = self.imgFood.shape
        self.foodpoints = 0,0
        self.randomfood()
        self.score = 0
        self.gameover = False

    def randomfood(self):
        self.foodpoints= random.randint(100,1000),random.randint(100,600)

    def update(self,imgMain, currentHead):
        if self.gameover:
            cvzone.putTextRect(imgMain, "Game Over", [300,400], scale= 4, thickness=4 , offset=20)
            cvzone.putTextRect(imgMain, f"Your Score:{self.score}", [300, 500], scale=4, thickness=4, offset=20)
        else:
            px,py = self.previoushead
            cx,cy = currentHead

            self.points.append([cx,cy])
            distance = math.hypot(cx-px,cy-py)
            self.lengths.append(distance)
            self.currentlength += distance
            self.previoushead = cx,cy

            #length reduction
            if self.currentlength > self.limitlength:
                for i, length in enumerate(self.lengths):
                    self.currentlength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentlength< self.limitlength:
                        break
            #check if snake eats food
            rx, ry = self.foodpoints
            w,h = self.wfood,self.hfood
            if rx - w//2 < cx < rx+w//2 and ry-h//2 < cy < ry + h//2:
                self.randomfood()
                self.limitlength +=50
                self.score +=1
                print(self.score)

            #draw snake
            if self.points:
                for i,point in enumerate(self.points):
                    if i!= 0:
                        cv2.line(imgMain, self.points[i-1],self.points[i],(0,0,255),20)
                cv2.circle(img, self.points[i-1], 10, (200,0,200), cv2.FILLED)
            #food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,(rx-self.wfood//2,ry-self.hfood//2))
            #score
            cvzone.putTextRect(imgMain, f"Your Score:{self.score}", [50, 80], scale=4, thickness=4, offset=20)

            #collision checking
            points = np.array(self.points[:-2], np.int32)
            points = points.reshape((-1,1,2))
            cv2.polylines(imgMain,[points], False,(0,200,0), 3)
            minDistance = cv2.pointPolygonTest(points,(cx,cy), True)

            if -1 < minDistance < 1:
                print("hit")
                self.gameover = True
                self.points = []  # all points
                self.lengths = []  # distance between each point
                self.currentlength = 0  # current length of the snake
                self.limitlength = 150  # total allowed length
                self.previoushead = 0, 0
                self.randomfood()
        return imgMain


game = SnakeGameClass("T1.png")
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img, flipType=False)
    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameover=False