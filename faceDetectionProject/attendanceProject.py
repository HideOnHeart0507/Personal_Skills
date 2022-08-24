import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
path = 'ImageAtt'
images = []
classnames = []
myList = os.listdir(path)
for cls in myList:
    currentImg = cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    classnames.append(os.path.splitext(cls)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dim = (400,600)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markATT(name):
    with open('att.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString =  now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facelocCF = face_recognition.face_locations(imgS)
    encodeCF = face_recognition.face_encodings(imgS,facelocCF)

    for encodeface, faceloc in zip(encodeCF, facelocCF):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        print(faceDis)
        matchindex = np.argmin(faceDis)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markATT(name)



    cv2.imshow('Webcam', img)
    cv2.waitKey(1)