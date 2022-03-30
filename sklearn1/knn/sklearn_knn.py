import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def img2vec(filename):
    # 将32x32的图片转化为1024的向量
    vec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vec[0,32*i+j] = int(lineStr[j])
    return vec

def handwritingClassTest():
    hwLabels =[]
    train = listdir('trainingDigits')
    lenthtrain = len(train)
    trainMat = np.zeros((lenthtrain,1024))
    for i in range(lenthtrain):
        filename = train[i]
        classNum = int(filename.split('_')[0])
        hwLabels.append(classNum)
        trainMat[i,:] = img2vec('trainingDigits/%s'%(filename))
    n = kNN(n_neighbors=3, algorithm='auto')
    n.fit(trainMat, hwLabels)
    error = 0.0
    lentest = len(listdir('testDigits'))
    test = listdir('testDigits')
    for i in range(lentest):
        filename = test[i]
        classNum = int(filename.split('_')[0])
        vec = img2vec('testDigits/%s'% (filename))
        result = n.predict(vec)
        if(result != classNum):
            error += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (error, error/lentest * 100))

if __name__ == '__main__':
    handwritingClassTest()