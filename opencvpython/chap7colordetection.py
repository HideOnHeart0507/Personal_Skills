import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    # & 输出一个 rows * cols 的矩阵（imgArray）
    print(rows,cols)
    # & 判断imgArray[0] 是不是一个list
    rowsAvailable = isinstance(imgArray[0], list)
    # & imgArray[][] 是什么意思呢？
    # & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
    # & 而shape[1]就是width，shape[0]是height，shape[2]是
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    # & 例如，我们可以展示一下是什么含义
    cv2.imshow("img", imgArray[0][1])

    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        # & 设置零矩阵
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("HueMinimum","Trackbars",0,179,empty)
cv2.createTrackbar("HueMaximum","Trackbars",179,179,empty)
cv2.createTrackbar("SatMinimum","Trackbars",0,255,empty)
cv2.createTrackbar("SatMaximum","Trackbars",255,255,empty)
cv2.createTrackbar("ValMinimum","Trackbars",0,255,empty)
cv2.createTrackbar("ValMaximum","Trackbars",255,255,empty)
path = 'pics/lambo.jpg'
while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("HueMinimum","Trackbars")
    hmax = cv2.getTrackbarPos("HueMaximum", "Trackbars")
    smin = cv2.getTrackbarPos("SatMinimum", "Trackbars")
    smax = cv2.getTrackbarPos("SatMaximum", "Trackbars")
    vmin = cv2.getTrackbarPos("ValMinimum", "Trackbars")
    vmax = cv2.getTrackbarPos("ValMaximum", "Trackbars")

    lower= np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgresult = cv2.bitwise_and(img,img,mask=mask)
    # cv2.imshow("origin",img)
    # cv2.imshow("originHSV",imgHSV)
    # cv2.imshow("mask",mask)
    # cv2.imshow("result",imgresult)
    imgstack = stackImages(0.6,([img,imgHSV],[mask,imgresult]))
    cv2.imshow("imgstack",imgstack)
    cv2.waitKey(1)