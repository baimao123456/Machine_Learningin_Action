# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:03:51 2018

@author: baimao
"""

from numpy import *
#导入数据
def loadDataSet(fileName ,delim='\t'):      #读取以tab键分割的数据
    dataMat = []                #assume last column is target value
    fr = open(fileName)         #打开文件
    for line in fr.readlines(): #读取每一行
#        curLine = line.strip().split('\t')#!!!官方代码错误 
        curLine = line.strip().split(delim)
#        fltLine = map(float,curLine) 这里不能用map，，Python2X返回的是list，，3.X返回的是map对象
        fltLine = [float(item) for item in curLine]# 将每行的内容保存成一组浮点数
        dataMat.append(fltLine)
    return mat(dataMat)
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)     #求平均值
    meanRemoved = dataMat - meanVals     #中心化，使得处理后的数据集的和为0
    covMat = cov(meanRemoved, rowvar=0)  #计算数据的协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算协方差矩阵的特征值和特征向量
    eigValInd = argsort(eigVals)               #将特征值按从小到大排序,f返回的是索引值
    eigValInd = eigValInd[:-(topNfeat+1):-1]   #从后往前按顺序选择topNfeat个特征，
    redEigVects = eigVects[:,eigValInd]        #选取topNfeat个特征向量，组成压缩矩阵
    lowDDataMat = meanRemoved * redEigVects    #lowDDataMat是数据在 topNfeat 个维度的空间的坐标的投影
    #利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #用lowDDataMat来重构 X，这个x才是降维后的
    return lowDDataMat, reconMat  #返回降维后的数据 和 利用压缩矩阵反构出的数据
#定义显示函数，显示降维后和降维钱的图像
def show(dataSet,lowDMat,reconMat):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,0].flatten().A[0],dataSet[:,1].flatten().A[0],
               marker = '^',s = 50)  #降维之前的图像分布
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],
               marker = 'o',s = 50,c = 'red') #降维之后的图像分布
#用平均值填充NaN值
def replaceNanWithMean(): 
    
    dataMat = []
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = shape(dataMat)[1]  #获得特征的个数
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i]) #求非NaN值的平均值
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal  #将NaN值替换成平均值
    return dataMat
if __name__ == "__main__":
    '''PCA  test
    dataSet = loadDataSet('testSet.txt')
    lowDMat ,reconMat = pca(dataSet,1)
    show(dataSet,lowDMat,reconMat)
    '''
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)     #求平均值
    meanRemoved = dataMat - meanVals     #中心化，使得处理后的数据集的和为0
    covMat = cov(meanRemoved, rowvar=0)  #计算数据的协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算协方差矩阵的特征值和特征向量
    print(eigVals)
    