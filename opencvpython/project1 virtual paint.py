import cv2
import numpy as np

cap = cv2.VideoCapture(0)
f_width= 640
f_height = 480
cap.set(3,f_width)
cap.set(4,f_height)
cap.set(10,150)

colorlist = [[5,90,255,21,255,255],
             # [145,92,81,179,255,255]
             # [129,94,108,179,255,255],
              [83,93,108,179,255,255],
             # [20,129,186,179,255,255],
              [20,60,133,66,255,255]
             ]
mycolorvalue = [[51,153,255],[255,0,0],[0,255,0]]

clist =['orange', 'purple', 'pink', 'blue', 'yellow', 'green']

mypoints = [] ##

def find_Color(img,colorlist,mycolorvalue):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints=[]
    for color in colorlist:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y= getContours(mask)
        cv2.circle(imgresult,(x,y),10,mycolorvalue[count],cv2.FILLED)

        # cv2.imshow(str(color[0]),mask)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1
    return newPoints

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #FIND OUTER
    x,y,w,h=0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgresult, cnt, -1, (255, 0, 0), 3)#draw blue bliones
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            # objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # cv2.rectangle(imgresult, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return x+w//2,y
    #tip of the pen

def draw(mypoints,mycolorvalue):
    for point in mypoints:
        cv2.circle(imgresult,(point[0],point[1]),10,mycolorvalue[point[2]],cv2.FILLED)

while True:
    success, img = cap.read()
    imgresult = img.copy()
    # cv2.imshow("video",img)
    newPoints= find_Color(img,colorlist,mycolorvalue)
    if len(newPoints) != 0:
        for newP in newPoints:
            mypoints.append(newP)
    if len(mypoints) != 0:
        draw(mypoints, mycolorvalue)
    cv2.imshow("videomask", imgresult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

