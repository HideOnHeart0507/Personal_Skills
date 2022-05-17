import cv2
img = cv2.imread('pics/touxiang.jpg')
print(img.shape)
cv2.imshow("Image",img)

#resize
imgresize = cv2.resize(img,(1000,500))
cv2.imshow("Image",imgresize)

#crop
imgcrop = img[0:200,200:500]
cv2.imshow("Image",imgcrop)

cv2.waitKey(0)