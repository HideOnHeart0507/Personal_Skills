import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

def img2vector(filename):
    returnVec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    testLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        testLabels.append(classNumber)
        
