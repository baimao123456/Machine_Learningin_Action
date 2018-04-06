R# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:11:39 2018

@author: baimao
"""
import numpy as np
import pandas as pd
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
#标准线性回归求解weight函数
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T  #xMat为输入的数据矩阵，，yMat为标签
    xTx = xMat.T*xMat             #求解矩阵
    if np.linalg.det(xTx) == 0.0:         #判断行列式是否为0，如果为0，不能求逆矩阵
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)   #ws为weight的一个最佳估计系数
    return ws
def predict_regress():
    xArr,yArr = loadDataSet('ex0.txt')  #导入数据
    ws = standRegres(xArr,yArr)         #求解最佳估计weight
    xMat = np.mat(xArr); yMat = np.mat(yArr)  #xMat为输入的数据矩阵，，yMat为标签
#    yHat = xMat*ws    #yHat为根据ws拟合的曲线，是一个估计值
    yHat = np.dot(xMat,ws)  #点乘，常见的矩阵乘法，，multiply是对应元素相乘
    #下面画出拟合曲线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)  #增加子图
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])#flatten可以将多维变成一维
    
    xCopy = xMat.copy()
    xCopy.sort(0)   #对xMat进行排序，，为了使坐标连续
    yHat = xCopy*ws   #计算估计值（为了画线的估计值，x是排序过后的）
    ax.plot(xCopy[:,1],yHat)  #画出回归线
    plt.show()
#     print(np.corrcoef(yHat.T,yMat))#测试值yHat和真实值yMat的相关程序，用来评价拟合程度的好坏
#局部加权线性回归，testPoint为预测点，，k控制权重的大小，权重越大说明周围的点的重要性越大，对生成的weight影响越大
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T  #xMat为输入的数据矩阵，，yMat为标签
    m = np.shape(xMat)[0]   #获得样本的个数
    weights = np.mat(np.eye((m)))  # 生成维度为m的对角矩阵
    for j in range(m):                      #生成权重矩阵
        diffMat = testPoint - xMat[j,:]     #计算预测点和其它点的距离
#        weights[j,j] = np.exp(np.square(diffMat*diffMat.T)/(-2.0*k**2))  #计算每个点的权重，这里的分子是距离，，回归线更为平滑
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))  #计算每个点的权重，这里的分子是距离的平方
    xTx = xMat.T * (weights * xMat)  
    if np.linalg.det(xTx) == 0.0: #判断行列式是否为0，如果为0，不能求逆矩阵
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat)) #计算该预测点的回归系数
    return testPoint * ws                    #返回相应的y的预测值  
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)  #获得样本的个数
    for i in range(m):  #对每一个点都进行一个回归过程，，得到m组回归系数
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)      #分别计算yHat
    return yHat 
#局部加权线性回归，，并画出图像
def lwlr_predict_plot():
    xArr,yArr = loadDataSet('ex0.txt')  #导入数据
#    xMat = np.mat(xArr); yMat = np.mat(yArr)  #xMat为输入的数据矩阵，，yMat为标签
    yHat = lwlrTest(xArr,xArr,yArr,0.02)  #通过局部线性加权回归得到的预测值
    xMat = np.mat(xArr); yMat = np.mat(yArr)  #xMat为输入的数据矩阵，，yMat为标签
    srtInd = xMat[:,1].argsort(0)  #获得排序的序号
#    xSort = xMat[srtInd][:,0,:]
    #下面画出拟合曲线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)  #增加子图
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s = 2,c = 'red')#flatten可以将多维变成一维
    #画出原始数据
    xCopy = xMat.copy()
    xCopy.sort(0)   #对xMat进行排序，，为了使坐标连续
    ax.plot(xCopy[:,1],yHat[srtInd])  #画出回归线，，这里必须用srtindex，，因为每个预测点有相应的预测值yHat，，xmat是排序过的
    plt.show()
#岭回归，，解决特征数比样本数多，，造成x不是满秩矩阵，无法求逆的问题，，
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat 
    denom = xTx + np.eye(np.shape(xMat)[1])*lam  #加上一个m*m的对角矩阵，使denom能够求解逆矩阵
    if np.linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)    #denom.I是求denom的逆矩阵，，ws为回归系数
    return ws
#岭回归测试，，这里需要对训练和测试数据标准化，，使得每维特征具有相同的重要性
def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    yMean = np.mean(yMat,0)  #求平均值
    yMat = yMat - yMean     #标准化数据
    #标准化 X's
    xMeans = np.mean(xMat,0)   #求平均值
    xVar = np.var(xMat,0)      #求方差
    xMat = (xMat - xMeans)/xVar #
    numTestPts = 30 #循环次数，，用来求解lam
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat
#返回平方误差
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()
#按列标准化数据，，使每个特征的重要性均衡
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis = 0)   #calc mean then subtract it off
    inVar = np.var(inMat,axis = 0)       #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat
#前向逐步回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = np.mat(xArr); yMat=np.mat(yArr).T  #转化为矩阵形式
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean         #数据标准化，0均值，单位方差
    xMat = regularize(xMat)
    m,n = np.shape(xMat)  #获得m为样本数，n为特征数
    returnMat = np.zeros((numIt,n)) #testing code remove
    ws = np.zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):   #进行numIt次迭代
        print (ws.T)
        lowestError = float("inf")  #设置当前最小误差为正无穷
        for j in range(n):  #针对每个特征(n个)
            for sign in [-1,1]:       #控制增大还是减小    
                wsTest = ws.copy()        #得到ws的一个副本
                wsTest[j] += eps*sign    #eps表示每次迭代需要的步长，就是每次增加或减少多少
                yTest = xMat*wsTest      #根据改变之后的weight计算预测值ytest，
                rssE = rssError(yMat.A,yTest.A)#计算更新后的weight得到的平方误差
                if rssE < lowestError:        #如果得到的误差比之前的最小误差还要小，则更新lowesterror
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat
##############################################################################
    #预测乐高玩具
##############################################################################
'''
from time import sleep
import json
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print ('problem with item %d' % i)

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)   
'''
#交叉验证10折，选取90%数据训练，，10%数据测试，，数据随机选取
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)     #获得样本的总个数                       
    indexList = list(range(m))  #获得一个范围为0-m的随机数list
    errorMat = np.zeros((numVal,30))    #建立一个10*30的矩阵，用来存储每次（一共10次）交叉验证的每组weight的误差
    for i in range(numVal):     #进行10次数据集的划分
        trainX=[]; trainY=[]
        testX = []; testY = []
        np.random.shuffle(indexList)    #对数据进行随机排序，，打乱顺序
        for j in range(m):              #indexList中的前 90%的数据作为训练集
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:   #indexList中的10%的数据作为测试集
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #岭回归通过每次划分的数据集（标准化），得到30组weight向量
        for k in range(30):     #遍历所有的30组weight，，利用标准化之后的数据计算平方误差
            matTestX = np.mat(testX); matTrainX=np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)#用训练时的参数标准化测试数据
            matTestX = (matTestX-meanTrain)/varTrain   #岭回归需要对数据特征进行标准化处理#测试集用与训练集相同的参数进行标准化  
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)     #岭回归的y的预测值
            errorMat[i,k]=rssError(yEst.T.A,np.array(testY))  #计算每i次交叉验证时，，第k组weight值的表现
            #print errorMat[i,k]
    meanErrors = np.mean(errorMat,0)    #计算10次交叉验证的每个lam生成的系数的平均误差,
    minMean = float(np.min(meanErrors))    #找到30个均值误差中最小的的那个
    bestWeights = wMat[np.nonzero(meanErrors==minMean)]  ##将均值误差最小的lam对应的回归系数作为最佳回归系数，，np.nonzero返回不为0元素的下标
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr); yMat=np.mat(yArr).T   #为了和standRegress（数据没有标准化）生成的系数作比较，，这里对bestweight做了还原
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights/varX  #标准化会除以方差，，求w时会反过来乘上它，，，所以这里再除以方差
    print ("the best model from Ridge Regression is:\n",unReg)
    print ("with constant term: ",-1*sum(np.multiply(meanX,unReg)) + np.mean(yMat))#常数项  y = ax+b-->b = -ax + y

def load_lego(setNum):
    htmlArr = 'setHtml/lego%d.html'%setNum
    htmlf = urllib.request.urlopen(htmlArr).read()
#    htmlcont=htmlf.read()
    print(htmlf)
 
if __name__ == "__main__":
    '''测试岭回归的系数随lam值的变化情况
    abX,abY = loadDataSet('abalone.txt')
    ridgeWeight = ridgeTest(abX,abY)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)  #增加子图
    ax.plot(ridgeWeight)
    plt.show()
    '''
    '''前向逐步回归算法测试，，，可以用来挑选特征
    abX,abY = loadDataSet('abalone.txt')
    stageWise(abX,abY,0.001,5000)
    '''
    abX,abY = loadDataSet('abalone.txt')
#    ws = standRegres(abX,abY)
    crossValidation(abX,abY)
#    setDataCollect(lgX,lgY)
