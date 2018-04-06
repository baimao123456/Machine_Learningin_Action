# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:02:47 2018

@author: baimao
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
def loadDataset():#返回的是数组形式形式，，（100,3）（100,1）
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  #将读取的一行分隔开
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#将labelMat[0]和labelMat[1]添加到dataSet中，并在第一行添加1.0
        labelMat.append([int(lineArr[2])])#将labelMat[0]添加到labelMat中，作为标签
    return dataMat,labelMat
def loadDataset_pandas(data):
   ones = pd.DataFrame({'ones':np.ones(len(data))})
   data = pd.concat([ones,data],axis = 1)#按列拼合在一起
   #return data.iloc[:,:-1].as_matrix()#iloc是按行索引，loc是按index序号索引
   return  data.iloc[:,:-1].as_matrix(),data['Y'].as_matrix().reshape(100,1)
def sigmoid(X):
    return 1.0/(1+np.exp(-X))
#梯度上升法，
def GrandAscent(X,Y):
    X = np.matrix(X)#将数组转化成矩阵
    Y = np.matrix(Y)
    
    m,n = X.shape[0],X.shape[1] #获得X向量的维度特征
    alpha = 0.001               #设置步长值，控制weight的收敛速度
    maxcycle = 500              #迭代次数
    weight = np.zeros((n,1))#初始化权重矩阵，，shape为（3,1），个数为样本特征的个数
    for i in range(maxcycle):
        h = sigmoid(X*weight)
        error = (Y - h)
        weight = weight + alpha*X.T*error
    return weight
def StoGrandAscent(X,Y):
    m,n = X.shape[0],X.shape[1] #获得X向量的维度特征
    alpha = 0.01               #设置步长值，控制weight的收敛速度
    weight = np.zeros(n)#初始化权重矩阵，，shape为（3,1），个数为样本特征的个数
    for i in range(200):
        for j in range(m):
            h = sigmoid(np.sum(X[i]*weight))
    #        print(h)
            error = Y[i] - h
            weight = weight + alpha*error*X[i]
    return weight
#随机梯度下降法
def SGD_UpGrade(X,Y,numIter=150):
    m,n = X.shape[0],X.shape[1] #获得X向量的维度特征
    weight = np.zeros(n)#初始化权重矩阵，，shape为（3,1），个数为样本特征的个数
    for i in range(numIter):    
        dataIndex = list(range(m))#生成一个100个数的list，，范围从0到m
        for j in range(m):
            alpha = 4/(1.0+i+j)+0.01  #设置步长值，控制weight的收敛速度，这里alpha是根据迭代步数进行相应调整，防止震荡
            randomIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(np.sum(X[randomIndex]*weight))
            error = Y[randomIndex] - h
            weight = weight + alpha*error*X[randomIndex]
            del(dataIndex[randomIndex])
    return weight
#画图函数
def plotBestFit():
    fig = plt.figure()#新建一个画布
    ax = fig.add_subplot(111)#增加一个子图
    plt.xlabel('X1');plt.ylabel('X2')#给子图的坐标轴命名
    ax.scatter(x=data['X1'].loc[data['Y']==0], y=data['X2'].loc[data['Y']==0], c='b')#画出Y=0的点，，画出Y=1的点
    ax.scatter(x=data['X1'].loc[data['Y']==1], y=data['X2'].loc[data['Y']==1], c='r')
    x = np.arange(-3.0,3.0,0.1)#设置x的区间（-3.0,3.0），步进值为0.1
    y = (-weight[0]-weight[1]*x)/weight[2] #这个式子根据0=（w0x0+w1x1+w2x2）,解出X1X2的关系式
    ax.plot(x,y.T)
    plt.show()
if __name__ == "__main__":
   data = pd.read_table('testSet.txt',names = ['X1','X2','Y'])
   X,Y = loadDataset_pandas(data)
   weight = SGD_UpGrade(X,Y)
   plotBestFit()
   