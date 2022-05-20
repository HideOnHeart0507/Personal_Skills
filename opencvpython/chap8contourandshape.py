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

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #FIND OUTER
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgcont, cnt, -1, (255, 0, 0), 3)#draw blue bliones
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            #
            # if objCor == 3:
            #     objectType = "Tri"
            # elif objCor == 4:
            #     aspRatio = w / float(h)
            #     if aspRatio > 0.98 and aspRatio < 1.03:
            #         objectType = "Square"
            #     else:
            #         objectType = "Rectangle"
            # elif objCor > 4:
            #     objectType = "Circles"
            # else:
            #     objectType = "None"
            #
            cv2.rectangle(imgcont, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(imgcont, objectType,
            #             (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            #             (0, 0, 0), 2)

path ='pics/cards.jpg'
img = cv2.imread(path)
imgcont = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
getContours(imgCanny)

imgblank = np.zeros_like(img)
imgstack = stackImages(0.8,([img,imgGray,imgBlur],
                            [imgCanny,imgcont,imgblank]))

cv2.imshow("stack",imgstack)

cv2.waitKey(0)