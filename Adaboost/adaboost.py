# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:29:36 2018

@author: baimao
"""
import numpy as np
import pandas as pd
def loadSimpData():
    dataMat = np.matrix([
            [1., 2.1],
            [2.,  1.1],
            [1.3,  1.],
            [1.,    1.],
            [2.,    1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels
#参数描述，        数据矩阵  维度(列)   分类阈值   不等式标识，>还是 <=
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] =  -1.0 #对于dataMatrix[:,dimen] <= threshVal的值，相应的位置上置为 -1
    else: 
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0  #对于dataMatrix[:,dimen] >  threshVal的值，相应的位置上置为 -1
    return retArray
def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = dataMatrix.shape  #m为样本的总数，，n为维度，也就是特征的个数
    numSteps = 10.0; bestStump = {} ;bestClasEst = np.mat(np.zeros((m,1))) 
    # numsteps为步进次数，次数越大，分类越细致，，bestStump存储各个参数值  bestClassEst存储最好的分类结果
    minError = float("inf")   #初始化错误率为无限大
    for i in range(n):  #对于数据集中的每个特征进行循环
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax - rangeMin)/numSteps  #获得根据最大最小值步长值
        for j in range(-1,int(numSteps)+1):  #对于步进值的循环，，来寻找最佳分类阈值
            for inequal in ['lt','gt']:  #对于每个不等号循环,用于判断
                threshVal = (rangeMin + float(j) * stepSize) #根据循环次数，确定判断阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))  #存储错误分类的个数矩阵
                errArr[predictedVals == labelMat] = 0  #如果分类错误，相应的位置上置 1，若正确则为 0
                weightedError = D.T*errArr  #计算总的分类错误率
               # print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"\
                     # % (i, threshVal, inequal, weightedError))
                if weightedError < minError:   #如果计算出来的总误差小于上次循环时的最小误差，，则更新它
                    minError = weightedError
                    bestClasEst = predictedVals.copy()  #最好的分类结果
                    bestStump['dim'] = i                #最好的分类维度，，这里指特征的选择（按什么分类）
                    bestStump['thresh'] = threshVal     #返回分类阈值
                    bestStump['ineq'] = inequal         #返回判断标准，是大于还是小于等于
    return bestStump,minError,bestClasEst
def adaBoostTrainDS(dataArr,classLabels,numIt =40):#默认迭代次数为40
    weakClassArr = []  #用于存放最佳单层决策树
#    m = dataArr.shape[0]
    m = np.shape(dataArr)[0]  #这里为了方便list没有shape函数，，所以用numpy的函数取得列值
    D = np.mat(np.ones((m,1))/m)  
    aggClassEst = np.mat(np.zeros((m,1)))     
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print('D:',D.T)
        alpha = float(0.5*np.log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  #保存最佳决策树
        print('classEst',classEst) #输出预测结果
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)#因为这里expon分类正确时为负值，错误时为正值，
        #这里classLabels乘以classEst，若分类正确，则为一个正m，若分类错误，则为-m
        #所以这里D要初始化时要除以m
        D = np.multiply(D,np.exp(expon)) #这里利用向量进行计算，，对每个i值进行操作，提高运算效率
        D = D/D.sum()  #更新D的值  ，这里除以sum（）是为了使D成为一个概率分布，
        aggClassEst += alpha*classEst  #类别估计累计值，，，正负表示分类类别，大于0为1，小于0为0
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst)!=  #sign用来分类，
                                np.mat(classLabels).T,np.ones((m,1))) #统计aggclassest中分类错误的个数
        errorRate = aggErrors.sum()/m  #计算错误率
        print("total error:",errorRate,"\n")
        if errorRate == 0:  break
    return weakClassArr,aggClassEst
#adaboost分类函数
def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):  #进行多次循环，将弱分类器按照alpha进行组合，得到一个强分类器
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst   #累计估计值，循环次数越多，越准确
        print (aggClassEst)
    return np.sign(aggClassEst)
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #得到特征的个数
    dataMat = []; labelMat = []
    fr = open(fileName)       
    for line in fr.readlines(): #读每一行
        lineArr =[]
        curLine = line.strip().split('\t')   #分割字符
        for i in range(numFeat-1):           #对每一行的每一个特征进行提取
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr) 
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #绘点的坐标
    ySum = 0.0 #计算AUC的变量
    numPosClas = sum(np.array(classLabels)==1.0) #正例的个数
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas) #x，y轴上的步长值
    sortedIndicies = predStrengths.argsort()#获取排序的索引
    fig = plt.figure()  #设置画图工具
    fig.clf()
    ax = plt.subplot(111)
    #遍历所有的值, 每个点之间画一条线！！这里不是很懂
    for index in sortedIndicies.tolist()[0]:  #对于排序好的每个索引值，
        if classLabels[index] == 1.0:         #如果相应的标签为1，则在y轴上下降一个步长，不断降低真阳率
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)  #xstep为小矩形的宽度，ysum为总的下降长度，故乘积为AOC的面积
if __name__ == "__main__":
    '''
    dataMat,classLabels = loadSimpData()
    classifierArr = adaBoostTrainDS(dataMat,classLabels,10)
    print(adaClassify([1,1],classifierArr))
    '''
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArr,aggClassEst = adaBoostTrainDS(dataArr,labelArr,10)
#    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
#    prediction10 = adaClassify(testArr,classifierArr)
#    errArr = np.mat(np.ones((67,1)))
#    print(errArr[prediction10 != np.mat(testLabelArr).T].sum())
    plotROC(aggClassEst.T,labelArr)