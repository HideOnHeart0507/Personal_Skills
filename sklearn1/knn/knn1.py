from matplotlib.font_manager import FontProperties
import matplotlib.lines as mines
import matplotlib.pyplot as plt

import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
    dataSetsize = dataSet.shape[0]
    diff = np.tile(inX,(dataSetsize,1)) - dataSet
    square_Dis = (diff**2).sum(axis=1)
    distance = square_Dis**0.5
    sorteddistindices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sorteddistindices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        print(sortedClassCount)
        return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename, 'r', encoding='utf-8')
    arrayLines = fr.readlines()
    arrayLines[0]=arrayLines[0].lstrip('\ufeff')
    number = len(arrayLines)
    returnMat =np.zeros((number,3))
    classLabelVector = []
    index = 0

    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

def showdatas(datingDataMat, datingLabels):
	#设置汉字格式
	font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)  ##需要查看自己的电脑是否会包含该字体
	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

	numberOfLabels = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')
	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
	plt.setp(axs1_title_text, size=9, weight='bold', color='red')
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
	plt.setp(axs2_title_text, size=9, weight='bold', color='red')
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
	#设置图例
	didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',
	                  markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',
	                  markersize=6, label='largeDoses')
	#添加图例
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
	#显示图片
	plt.show()

def autoNorm(dataSet):
	#获得数据的最小值
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	#最大值和最小值的范围
	ranges = maxVals - minVals
	#shape(dataSet)返回dataSet的矩阵行列数
	normDataSet = np.zeros(np.shape(dataSet))
	#返回dataSet的行数
	m = dataSet.shape[0]
	#原始值减去最小值
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	#除以最大和最小值的差,得到归一化数据
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	#返回归一化数据结果,数据范围,最小值
	return normDataSet, ranges, minVals

def datingClassTest():
	#打开的文件名
	filename = "datingTestSet.txt"
	#将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
	datingDataMat, datingLabels = file2matrix(filename)
	#取所有数据的百分之十
	hoRatio = 0.10
	#数据归一化,返回归一化后的矩阵,数据范围,数据最小值
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#获得normMat的行数
	m = normMat.shape[0]
	#百分之十的测试数据的个数
	numTestVecs = int(m * hoRatio)
	#分类错误计数
	errorCount = 0.0

	for i in range(numTestVecs):
		#前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
			datingLabels[numTestVecs:m], 4)
		print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))

def classifyPerson():
	#输出结果
	resultList = ['讨厌','有些喜欢','非常喜欢']
	#三维特征用户输入
	precentTats = float(input("玩视频游戏所耗时间百分比:"))
	ffMiles = float(input("每年获得的飞行常客里程数:"))
	iceCream = float(input("每周消费的冰激淋公升数:"))
	#打开的文件名
	filename = "datingTestSet.txt"
	#打开并处理数据
	datingDataMat, datingLabels = file2matrix(filename)
	#训练集归一化
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#生成NumPy数组,测试集
	inArr = np.array([ffMiles, precentTats, iceCream])
	#测试集归一化
	norminArr = (inArr - minVals) / ranges
	#返回分类结果
	classifierResult = classify0(norminArr, normMat, datingLabels, 3)
	#打印结果
	print("你可能%s这个人" % (resultList[classifierResult-1]))

if __name__ == '__main__':
	datingClassTest()