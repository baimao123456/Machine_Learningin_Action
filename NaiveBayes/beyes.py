# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:31:43 2018

@author: baimao
"""
import numpy as np
from numpy import *
import re
import feedparser
def loadDataSet():#构造一个词表
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
def createVocabList(dataSet):#获得一个词汇list，里边的元素互不相同
    vocabSet = set([])   #获得一个set类型的变量
    for document in dataSet:  #遍历dataSet，求并集
        vocabSet = vocabSet | set(document)#创建两个集合的并集
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):#获得词向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:  
        if word in vocabList:  #判断imput中的词，是否在集合里边
            returnVec[vocabList.index(word)] = 1   #查找word在vocabList中的位置，如果存在则在位置上置1
#            print(vocabList.index(word))
        else: print('the word: %s is not in my Vocabulary!'%word)
    return  returnVec
#朴素贝叶斯分类器训练函数，输入参数trainMatrix为文档矩阵，trainCategory为每篇文档类别标签labels（1或者0）
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  #训练集所有文档的个数
    numWords = len(trainMatrix[0])   #每篇文档中，词的个数 
    pAbusive = np.sum(trainCategory)/float(numTrainDocs) #计算P（ci=1）的概率 
   # p0Num = np.zeros(numWords);p1Num = np.zeros(numWords)  #构造两个数组，用来存放每个元素的概率
  
#    p0Denom = 0.0;p1Denom = 0.0
    p0Num = np.ones(numWords);p1Num = np.ones(numWords) #为了防止概率相乘为0，这里初始化为1
    p0Denom = 2.0;p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  #对于ci=1，
            p1Num += trainMatrix[i] #这里是两个向量相加，用来统计类别1中，各个词出现的个数
            p1Denom += np.sum(trainMatrix[i])  #因为上一步是向量相加，所以这里要增加所有词条的计数值
        else:  #对于ci=0， 
            p0Num += trainMatrix[i]  #这里是两个向量相加，用来统计类别1中，各个词出现的个数
            p0Denom += np.sum(trainMatrix[i]) #因为上一步是向量相加，所以这里要增加所有词条的计数值
    p1Vect = np.log(p1Num/p1Denom)    #用每个元素/总词条的数目得到条件概率，，返回的是任意文档属于ci=1的概率，最大概率的元素影响因子最大
    p0Vect = np.log(p0Num/p0Denom)  #这里用log函数，因为log函数和f（x）函数具有相同的单调性，为了避免数过于小，而约成0
    print((p1Vect))
    return p0Vect,p1Vect,pAbusive
def classifyNB(vect2Classify,p0Vec,p1Vec,pClass1):
    p1 = np.sum(vect2Classify * p1Vec) + np.log(pClass1)  #将vect2Classify在词向量中所对应的概率相加，，求得分类概率
    p0 = np.sum(vect2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)  #获得一个词的集合，每个词出现一次
    trainMat = []#用来存放每个文档的词向量，一共6个文档
    for postingDoc in listOPosts:  #将每个文档依次转化成词向量，并保存在trainMat中
        trainMat.append(setOfWords2Vec(myVocabList,postingDoc))
    p0v,p1v,pAb = trainNB0(np.array(trainMat),np.array(listClasses))#训练分类器，
    
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0v,p1v,pAb))
    
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0v,p1v,pAb))
#因为一个词在文档中出现的次数不止一次，所以改进一下setOfWords2Vec，，成为词袋模型
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:  
        if word in vocabList:  #判断imput中的词，是否在集合里边
            returnVec[vocabList.index(word)] += 1   #查找word在vocabList中的位置，如果存在则在位置上+1
    return  returnVec
def textParse(bigString):
     listOfTokens = re.split(r'\W*',bigString)
     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
 #垃圾邮件测试
def spamTest():
    docList =[];classList =[];fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' %i).read()) #读取文本文件，并分割文本并提取数据 
        docList.append(wordList)
        fullText.extend(wordList) #这个是把文本文档中所有元素都拉出来，做一个大向量
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %i).read())#第23个文件多了个？号，导致编译错误
        docList.append(wordList)
        fullText.extend(wordList) #这个是把文本文档中所有元素都拉出来，做一个大向量
        classList.append(0)
    vocabList = createVocabList(docList)  #获得一个词典，，里边包括所有出现的单词
    trainingSet = list(range(50)); testSet = []  #trainingSet为50个随机数
    for i in range(10):  # 随机选择10个文档,包括spam和ham
        rangeIndex = int(np.random.uniform(0,len(trainingSet))) #random.uniform(x, y)生成下一个随机数，包括x，不包括y
        testSet.append(trainingSet[rangeIndex])  # 获得一个随机数的list，里边的序号为测试集
        del trainingSet[rangeIndex]  # 删除选中的索引值，防止出现重复数,剩下的数作为训练集
        trainMat = []   # 用来存放每个文档的词向量，一共6个文档
        trainClasses = []

    for docIndex in trainingSet:  #将每个文档依次转化成词向量，并保存在trainMat中
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))#训练分类器，d得到各个词对分类的影响概率
    print(p0v)
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])  #对每封测试邮件构建词向量
        if classifyNB(np.array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:  #如果和实际标签不一致，错误加1
               errorCount += 1
               print(docList[docIndex])
    print('the error rate is:',float(errorCount/len(testSet)))
def calMostFreq(vocabList,fullText):#统计出现单词频率最高的
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key =operator.itemgetter(1),reversed = True)
    #sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
    #list 的 sort 方法返回的是对已经存在的列表进行操作，
    #而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
    #revers为true则为降序排列
    return sortedFreq[:30]   #返回排序最高的30个单词

def localWords(feed1,feed0):
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):  
        wordList = textParse(feed1['entries'][i]['summary'])  #获取每篇文章的内容
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])  #获取每篇文章的内容
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
#    vocabList = createVocabList(docList)
        print(wordList)
    
if __name__ == "__main__":
#   ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss') 
#   sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss') 
#    NASA Image of the Day：http://www.nasa.gov/rss/dyn/image_of_the_day.rss
   ni = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')  #一共9篇文章
#　　Yahoo Sports - NBA - Houston Rockets News：http://sports.yahoo.com/nba/teams/hou/rss.xml
   ys = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
#   tx = feedparser.parse('http://news.qq.com/newsgn/rss_newsgn.xml')
#   print(ni.keys()) 
#   print((ni['entries'])[1]['summary'])
   localWords(ni,ys)
    
    