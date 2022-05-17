import cv2
import numpy as np
img = cv2.imread("pics/cards.jpg")

width,height =250,350
pts = np.float32([[1115-903,197-135],[1361-903,256-135],[1015-903,577-135],[1278-903,634-135]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("output",img)
cv2.imshow("new",imgOutput)
cv2.waitKey(0)