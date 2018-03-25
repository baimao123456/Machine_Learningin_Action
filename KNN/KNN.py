# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:01:00 2018
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label
@author: baimao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os
def CreatDataSet(): #创建一个用于测试的数组
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
#分类器
def classify0(inX,dataSet,labels,k):#inX,shape = (1,3),dataSet(1000,3)label(1000,1)
    dataSetSize = dataSet.shape[0]
#    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet  #np.tile的作用是复制dataSetSize份,拓展向量以便和每个点的坐标相减
#    sqdiffMat = diffMat**2  #diffMat的平方
#    sqDistances = sqdiffMat.sum(axis =1)  #平方和
    distances = np.sqrt(np.sum((inX-dataSet)**2,axis=1))#求距离公式，比较简便的写法
    sortedDisIndicies = distances.argsort()#按x的值排序，并给出相应的序号，比如[2,1,3,4]->[1,0,2,3]
    classCount = [] #用于存放匹配出来的labels
    for i in range(k):  #用于读取前k个label，k为最近的点
        votelabel = labels[sortedDisIndicies[i]  #读取第i个数据所对应的的labels
        classCount.append(votelabel)      
    c = Counter(classCount)     #调用counter函数，，计算classCount中出现label最多的，，作为分类的标签
    return c.most_common(1)[0][0]

def file2matrix(filename):      #datingTestSet.txt
    data = pd.read_table(filename,names = ['X1','X2','X3','Y'])
    return data.iloc[:,:-1].as_matrix(),data.iloc[:,-1].as_matrix()     #返回数据，，前三列是特征，，最后一列是lablel
def show():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    plt.show()
    sns.lmplot(x="X1", y="X2", hue="Y",
               fit_reg=False, size=5, data=data)  #画图，fit_reg如果为false，则不会画出线性回归线
def autoNorm(dataSet):      #归一化处理，，(max-min)/ranges
    minVals = dataSet.min(0)    #获得最小值
    maxVals = dataSet.max(0)    #获得最大值
    ranges = maxVals - minVals   #获得值的范围
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet -minVals
    normDataSet = normDataSet/ranges
    return normDataSet,ranges,minVals 
def datingClassTest():  #这里1-100作为测试集，，100-1000作为训练集（计算这100个样本距离那900个点的距离，从中选取离得最近的3个，其标签作为预测值）
    hoRatio = 0.10      #此系数控制测试样本选取的比例
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals =  autoNorm(datingDataMat)
    m = normMat.shape[0]    #样本的总数
    numTestVecs = int(m*hoRatio)    #测试的样本数，这里是100
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],#array[m:n,:]表示取m到n行，每列都取，，array[；,m:n]表示取m到n列，每行都取
           datingLabels[numTestVecs:m],3)
        print("the %d classifierResult came back with %d,the real answer is %d"
              %(i,classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):errorCount += 1.0  #如果预测结果不一致，计数错误
    print("the total error rate is%f"%(errorCount/float(numTestVecs)))#注意此处需要用%f，，因为输出的是个浮点数
def classifyPerson():
    resultList = ['not at all','a small does','a large does']
    percentTats = float(input("percent of time spent playing video games?"))#input(),允许用户输入文本命令并返回用户输入的命令
    ffMiles = float(input("frequent filter miles earned per years?"))
    iceCream = float(input("liters of icecream consumed per year?"))
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals =  autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])#构造一个用来预测的样本，将输入的数据组合起来
    clssifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)#特别注意，在用自己输入的数据进行预测时，一定得归一化，，否则会出错
    print("you will probably like this person: ",resultList[clssifierResult-1])
def img2vector(filename):
    returnVect = np.zeros((1,1024))#构造一个（1,1024）矩阵，用来存放向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#将读出的字符转换成数字，并存放在矩阵中
    return returnVect
def handwritingClassTest():
    hwLables = []   #存储training中真实的标签
    trainingFileList = os.listdir('trainingDigits')#获取trainingDigits下边的目录内容
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))   #构造一个（m，1024）的矩阵，存放m个样本
    for i in range(m):
        fileNameStr = trainingFileList[i]  #获得每个文件的名字，比如0_13.txt
        fileStr = fileNameStr.split('.')[0]#去掉文件类型后缀，获得比如0_13
        classNum = int(fileStr.split('_')[0])#获得文件中保存的数字，，比如0
        hwLables.append(classNum)            #将label值加入到hwlabels中
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)#将图片转换成矩阵形式，格式为（1,1024）
    #获得labels
    testFileList = os.listdir('testDigits') #获取teatDigit下边的目录内容
    errorCount = 0.0
    mTest = len(testFileList)#获得test集中的样本个数
    for i in range(mTest):
        fileNameStr = testFileList[i]  #获得每个文件的名字，比如0_13.txt
        fileStr = fileNameStr.split('.')[0]#去掉文件类型后缀，获得比如0_13
        classNum = int(fileStr.split('_')[0])#获得文件中保存的数字，，比如0
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLables,3)
        print("the classifier come back with: %d,the real answer is: %d"%(classifierResult,classNum))
        if(classifierResult != classNum):  errorCount += 1.0 #如果不一样，则errorCount加1
        
        
    print("\nthe total number of error is:%d"%errorCount)#分类错误总数
    print("\nthe total error rate is:%f"%(errorCount/float(mTest)))#分类错误率
    
if __name__=="__main__":
    '''group,labels = CreatDataSet()
    diffMat = np.tile([0,0],(4,1)) - group
    index = (classify0([1,0],group,labels,3))
    print(index)'''
#    datingClassTest()#约会分类器
#    classifyPerson()#根据特征预测对这个人的喜欢程度
    vect = img2vector('testDigits/0_13.txt')
    handwritingClassTest()
    
    
    