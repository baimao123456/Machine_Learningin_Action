# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 09:08:52 2018
SVM 支持向量机   SMO算法
@author: baimao
"""
from numpy import *
from time import sleep

#读取数据（特征和labels）
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
#返回一个和 i不同的随机值，，m是所有alpha的数目
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j
#用于调整大于H和小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj
#简化版的SMO
def smoSimple(dataMatIn, classLabels, C, toler, maxIter): #toler为容错率，maxIter为最大循环次数
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)  #m，n分别是特征数和样本的总数
    alphas = mat(zeros((m,1)))      #alpha和样本的个数相同
    iter = 0
    while (iter < maxIter):         #
        alphaPairsChanged = 0       #用于标志 alpha是否被优化
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b #为预测的类别
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < - toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b #预测的类别
                Ej = fXj - float(labelMat[j])   #真实值和预测值之间的误差
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy(); #旧的alpha值
                if (labelMat[i] != labelMat[j]):  #
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:print("L==H"); continue  #不做任何改变
                #eta是最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta  #统计方法学7.106，这里符号不一样，但是不影响使用
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue  #如果alpha没有有轻微改变，则继续下一循环
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#i和j进行相同的改变，但是改变得方向相反，一个增加一个减小
                                                                      
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1  #满足 0 < alpha < C
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2 #满足 0 < alpha < C
                else: b = (b1 + b2)/2.0 #如果alpha1_new和 alpha2_new 是0或者C,b1_new和b2_new之间的值都符合要求，则取中间值
                alphaPairsChanged += 1 #优化了一次
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1 #如果没有进行优化，则继续迭代，寻找优化
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas
#核函数,,kTup为一个元组，，（核函数类型，theta值）
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #线性核
    elif kTup[0]=='rbf':             #高斯核
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K
#构建一个optStruct类，用来保存数据
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn           #特征数据
        self.labelMat = classLabels  #类别标签
        self.C = C                   #常数C
        self.tol = toler             #toler为容错率
        self.m = shape(dataMatIn)[0] #获取样本的个数
        self.alphas = mat(zeros((self.m,1))) #系数alpha 待求
        self.b = 0                           #系数b 待求
        self.eCache = mat(zeros((self.m,2))) #第一列标识了ecache是否有效的标志位，第二列是误差
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
#计算误差       
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
#选择第二个alpha，根据使误差变化最大的       
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1: #如果不是第一次循环，也就是已经有了一个alpha值
        for k in validEcacheList:  #loop through valid Ecache values 找一个使得最大化减小E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #如果是第一次循环，随机找一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
#更新误差值
def updateEk(oS, k):#任何一个alpha改变了，更新相应的误差值
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
#寻找决策边界的优化例程,,跟简化版本的差不多，区别在于第二个alpha选择算法不一样，不是随机选择的
#如果有任意一对alpha发生改变，则会返回1
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0  #如果界相同，则返回0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #这里改成了带核函数kenerl的
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)   #对alpha[j]进行剪辑
        updateEk(oS, j) #更新误差列表
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0#如果改变得太小，返回0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#和j更新相同的值，，但是是相反的方向
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0
#完整版Platt Smo算法
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0 #迭代数为0
    entireSet = True; alphaPairsChanged = 0  #entireSet控制是全部遍历还是仅遍历无边界值，当第一个alpha选择后，就会置1
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)): #如果迭代次数超多最大值，或者此次循环并没有改变alpha值，就退出循环 
        alphaPairsChanged = 0  #alpha改变与否标志位
        if entireSet:   #遍历所有的可能的alpha
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:           #遍历非边界值alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] #找到非边界alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas
#计算W值
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
#径向基函数测试
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt') 
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]   #alpha > 0的点为支持向量的索引
    sVs=datMat[svInd]   #得到支持向量
    labelSV = labelMat[svInd];  #得到支持向量对应的labels
    print("there are %d Support Vectors" % shape(sVs)[0]) #打印支持向量的个数
    #训练模型
    m,n = shape(datMat)
    errorCount = 0     #统计错误
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    #加载测试数据
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')  
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))  
#图片转向量函数
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
#加载图片
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    
#数字图像识别
def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages("trainingDigits")
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m)) 
if __name__ == "__main__":
    dataArr,labelArr = loadDataSet('testSet.txt')
    b,alpha = smoP(dataArr,labelArr,0.6,0.001,40)
    ws = calcWs(alpha,dataArr,labelArr)
    print(sign(mat(dataArr)[0]*mat(ws) + b))
    testRbf()
    testDigits(('rbf', 10))