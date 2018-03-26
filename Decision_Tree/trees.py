# -*- coding: utf-8 -*-
"""
Spyder Editor
决策树 采用id3来计算信息增益，有缺陷
This is a temporary script file.
"""
import numpy as np
from math import log
from collections import Counter
from treePlotter import *
import pickle
def calcShannonEnt(dataSet):#计算信息熵
    numEntries = len(dataSet)    #获得数据的长度
    labelCounts = {}            #用来存储label的dict
    for featVec in dataSet:    #一行行读取
        currentLabel = featVec[-1]  #标签，只有一个
        if currentLabel not in labelCounts.keys():#统计每个样本所属种类的个数
            labelCounts[currentLabel] = 0      #如果没有此类型，则为0 
        labelCounts[currentLabel] += 1   #如果在的话计数加1
    shannonEnt = 0.0 #初始化信息熵为0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries#所占的比例
        shannonEnt -= prob*log(prob,2)    #信息熵    
    return shannonEnt
def creatDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']
              ]
    labels = ['no surfacing','flippers']
    return dataSet,labels
def splitDataSet(dataSet,axis,value):#按照特定特征分割数据集：dataet:待划分的数据集，axis为划分数据集的属性（列号），value为用属性的哪个特征划分
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])#取后两列的值
            retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)   #信息熵
    bestInfoGain = 0.0;bestFeature = -1  #最佳信息增益和最好的特征值
    for i in range(numFeatures):
        featlist = [example[i] for example in dataSet]  #获得属性值列表
        uniqueVals = set(featlist)  #获得唯一的属性值，set中存储的是互不相同的元素
        newEntropy = 0.0  #定义一个存储按每个属性划分的信息熵，不停改变i*value次
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):#如果得到的信息增益比之前最好的要好，更新bestinfogain
            bestInfoGain = infoGain
            bestFeature = i          #更新相应的bsetfeature，i对应的属性 能够获得最大信息增益
    return bestFeature
def majorityCnt(classList):   
    c = Counter(classList)     #调用counter函数，，计算classCount中出现label最多的，，作为分类的标签
    return c.most_common(1)[0][0]
def createTree(dataSet,labels):
#    print(dataSet,"dataset")
    classList = [example[-1] for example in dataSet]  #获得labels列表最后一列,这里是所有的类标签，yes或者no
    if classList.count(classList[0]) == len(classList):#如果类别完全相同，则停止划分，classList[0]位样本的个数
        return classList[0]    #这里采用比较classLebel[0]和classLabel的长度，，若相等，说明这些特征属于同一个
    if len(dataSet[0]) == 1:#遍历完所有特征时，返回出现次数最多的
        return majorityCnt(classList)
    ###########以上为迭代的停止条件###################
    bestFeat = chooseBestFeatureToSplit(dataSet)  #选择一个最好的分类特征,是一个int型数，索引
    bestFeatLabel = labels[bestFeat]  #通过索引获得特征，，这里是字符串
    myTree = {bestFeatLabel:{}}  #构造第一个分类结点,之后每次迭代都会生成一个结点
    del (labels[bestFeat])  #删除已经用来分类的特征
    featValues = [example[bestFeat] for example in dataSet]#获得当前特征的属性值列表
    uniqueVals = set(featValues) #获得唯一的属性值，set中存储的是互不相同的元素
    for value in uniqueVals:   #遍历特征中的属性值
        subLabels = labels[:]  #复制标签列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree                      #在每个划分的数据集上生成树，这里是迭代的重点
def classify(inputTree,featLabels,testVec): 
    firstStr = list(inputTree.keys())[0]   #获得父节点的名字
    secondDict = inputTree[firstStr]       #子树
    featIndex = featLabels.index(firstStr) #将标签字符串转换成索引，找到firstStr所处的位置，以便于比较
    for key in secondDict.keys():          #遍历特征，
        if testVec[featIndex] == key:      #如果两个值相等，则选择一个分支，继续迭代，直到返回标签
            if type(secondDict[key]).__name__ == 'dict':#如果不是叶子节点，继续迭代
                classLabel = classify(secondDict[key],featLabels,testVec) #如果遇到父节点（判断节点），，继续迭代
            else: classLabel = secondDict[key]   #如果是叶子节点，返回标签
    return classLabel
def storeTree(inputTree,filename):#存储决策树，，书上的例子不对
    fw = open(filename,'wb')   #必须以二进制可写形式新建一个文件
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):            #读取决策树
    fr = open(filename,'rb')  #必须以二进制可读形式读入一个文件
    return pickle.load(fr)
def classifier_lenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]   #'\t'为水平制表符，用来分割数据
    lensesLabels = ['age','prescript','astigmatic','tearRate']    #创建标签列表
    lenseTree = createTree(lenses,lensesLabels)  #创建树
    createPlot(lenseTree)
    
if __name__ == "__main__":
    '''
    myDat,labels = creatDataSet()
    myTree = createTree(myDat,labels)
#    myTree = retrieveTree(1)
    storeTree(myTree,'classifierStorage.txt') #文件名后缀  txt和pkl都可以
    print(grabTree('classifierStorage.txt'))
    '''
    classifier_lenses()  #隐形眼镜预测
    
    
    
    
    