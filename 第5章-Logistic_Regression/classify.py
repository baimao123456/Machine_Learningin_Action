# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns

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
#定义分类函数
def ClassfyVector(X,weight):
    pass
#计算分类误差
def predict(weight, X):
    probability = sigmod(np.sum(X * weight,axis=1))
    return [1 if x >= 0.5 else 0 for x in probability]
    return probability
#计算分类错误率
def ColiTest():  
    train = pd.read_table('horseColicTraining.txt')
    test = pd.read_table('horseColicTest.txt')
   
    trainSet = train.iloc[:,:-1].as_matrix()     #train的样本
    train_labal = train.iloc[:,21:22].as_matrix() #train的标签,一共22列，取最后一列
    testSet = test.iloc[:,:-1].as_matrix()      #test的样本
    test_labal = test.iloc[:,21:22].as_matrix() #test的标签
    
    columns = train.shape[1]  #获得列数
    numtest = test.shape[0]   #获得test的行数，，也是样本的数量
    
    trainweight = SGD_UpGrade(trainSet,train_labal,200)#用随机梯度下降法进行模型的训练
    
    predict_value = predict(trainweight,testSet)#通过训练的weight预测的值
    pre_error = predict_value-test_labal.T
    error_rate = np.sum(pre_error==0)/numtest#统计0的个数，，若预测准确，则两个数相减为0
    return 1.0 - error_rate  #错误率
def MultiTest():
    numTests = 10;  errorSum = 0.0
    for k in range(numTests):
        errorSum += ColiTest()
        print("after %d iterations the average error tate is :%f" %(numTests,errorSum/numTests))
if __name__ == "__main__":
    MultiTest()
