import numpy as np
import cv2
import face_recognition

imgGuchi = face_recognition.load_image_file('image/guchi.jpg')
imgGuchi = cv2.cvtColor(imgGuchi, cv2.COLOR_BGR2RGB)
imgGuchiT = face_recognition.load_image_file('image/guchiT.jpg')
imgGuchiT = cv2.cvtColor(imgGuchiT, cv2.COLOR_BGR2RGB)
dim = (400, 600)

# resize image
imgGuchi = cv2.resize(imgGuchi, dim, interpolation=cv2.INTER_AREA)
imgGuchiT = cv2.resize(imgGuchiT, dim, interpolation=cv2.INTER_AREA)
print(imgGuchi)
faceloc = face_recognition.face_locations(imgGuchi)[0]
encodeGuchi = face_recognition.face_encodings(imgGuchi)[0]
cv2.rectangle(imgGuchi, (faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255), 2)

facelocTest = face_recognition.face_locations(imgGuchiT)[0]
encodeTest = face_recognition.face_encodings(imgGuchiT)[0]
cv2.rectangle(imgGuchiT, (facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255), 2)

results =face_recognition.compare_faces([encodeGuchi],encodeTest)
facedist = face_recognition.face_distance([encodeGuchi],encodeTest)
cv2.putText(imgGuchiT, f'{results}{round(facedist[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
print(results)

cv2.imshow('Guchi', imgGuchi)
cv2.imshow('GuchiTest', imgGuchiT)
cv2.waitKey(0)