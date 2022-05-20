import cv2
import numpy as np
print("Package imported")
# 图片
# img = cv2.imread("pics/touxiang.jpg")
#
# cv2.imshow("output", img)
# cv2.waitKey(0)

# 视频
# cap = cv2.VideoCapture("video/test.mp4")
# while True:
#     success, img = cap.read()
#     cv2.imshow("video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    success, img = cap.read()
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# img = cv2.imread('pics/touxiang.jpg')
# kernel = np.ones((5,5), np.uint8)
#
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)
# imgCanny = cv2.Canny(img, 100, 100) #找线条
# imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)#线条加粗
# imgEroded = cv2.erode(imgDilation, kernel, iterations=1)#线条变细
#
# # cv2.imshow("original", img)
# # cv2.imshow("gray",imgGray)
# # cv2.imshow("blur",imgBlur)
# # cv2.imshow("canny",imgCanny)
# cv2.imshow("dilation", imgDilation)
# cv2.imshow("eroded", imgEroded)
cv2.waitKey(0)