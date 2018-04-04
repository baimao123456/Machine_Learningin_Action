# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:46:07 2018

@author: baimao
"""
from numpy import *
#导入数据
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
#构建一个候选集的列表[frozenset({5}), frozenset({2}), frozenset({4}), frozenset({3}), frozenset({1})]
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:  #如果C1中没有item，则将item添加到C1中，保证C1中的元素各不相同
                C1.append([item])  #添加包括item元素的列表
    #map() 会根据提供的函数对指定序列做映射，，这里使用frozenset对C1进行操作
    return list(map(frozenset, C1))#frozenset是冻结的集合，它是不可变的，存在哈希值
                                    #好处是它可以作为字典的key，也可以作为其它集合的元素   
                                    #注意在Python3.X版本，，map返回的不是list，而是iterater，所以需要转换成list
# 得到支持度大于minsupport的候选集，#D:数据集，Ck候选项集列表，minSopport最小支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}  # 存放每个候选集在整个交易记录中的出现次数，，用来计算支持度
    for tid in D:       # 遍历所有交易记录
        for can in Ck:  # 遍历所有候选集
            if can.issubset(tid):  # 如果候选集中某项是tid的子集
#                if not ssCnt.values(can): ssCnt[can]=1 # 如果ssCnt中没有 can项，则将这一项置1
                if not ssCnt.__contains__(can): ssCnt[can]=1  #python3.X
                else: ssCnt[can] += 1     # 如果ssCnt中有can项，，直接自加1
    numItems = float(len(D)) # 获得总的交易记录数
    retList = [] 
    supportData = {}  # 支持度集合，键：frozenset({1})，码为支持度的值
    for key in ssCnt: # 计算每一项的支持度
        support = ssCnt[key]/numItems   #ssCnt[key] 对应着每个候选集在整个交易记录中的出现次数
        if support >= minSupport:    #寻找最大支持度
            retList.insert(0,key)    #如果支持度大于 minSupport，，存放到retlist
        supportData[key] = support
    return retList, supportData   #retlist是符合支持度大于minSupport的集合
                                  #supportData是带支持度的dict字典
#根据候选集LK中的k个元素构建集合，k控制用几个元素构建集合
def aprioriGen(Lk, k): #!!!creates Ck,,当LK中只有一个集合时，返回空[]
    retList = []
    lenLk = len(Lk)  #计算LK中元素数目
    for i in range(lenLk): #将每个集合和其它集合作比较
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]#!!!减少了遍历列表的次数
            L1.sort(); L2.sort()
            if L1==L2: #如果k-2相同
                retList.append(Lk[i] | Lk[j]) #set求并
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)  #获得候选集
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2  #用两个元素组成新的元素
    while (len(L[k-2]) > 0):  #只要LK不为空，一直循环
        Ck = aprioriGen(L[k-2], k)  #对L[k-2]的元素两两组合，，Ck为组合后的
        Lk, supK = scanD(D, Ck, minSupport)# 找到ck中满足支持度的集合LK
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
#生成关联规则list
def generateRules(L, supportData, minConf=0.7):  #supportData是一个来自频繁项集的dict
    bigRuleList = []
    for i in range(1, len(L)):#从1开始，是因为必须需要有至少两个元素的项集开始构建过程
        for freqSet in L[i]:  #对于每一个频繁项集 
            H1 = [frozenset([item]) for item in freqSet]# 创建只包括单个元素集合的列表H1
            if (i > 1): #如果i从2开始，说明频繁项集的元素超过2，进行合并处理
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: #如果频繁项集只有两个元素，直接计算可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         
#计算可信度，freqSet：频繁项集，H：只包括单个元素集合的列表，brl存储规则--可信度的list，，minConf最小可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] ##返回一个满足最小可信度要求的规则列表
#    print(freqSet,'fre')
    for conseq in H:  #对于频繁项集中的每个元素,求规则（其他元素-->这个元素）的可信度
        conf = supportData[freqSet]/supportData[freqSet-conseq] #计算可信度，，supportData[freqSet-conseq]为求集合的差，就是求其他的元素
        if conf >= minConf:   #如果可信度大于最小可信度，输出该规则
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
#如果频繁项集的元素数目超过2，进一步合并,#参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])  #获得H中的频繁项集的元素个数
    if (len(freqSet) > (m + 1)): #尝试合并
        Hmp1 = aprioriGen(H, m+1)##存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        if (len(Hmp1) > 1):#满足最小可信度要求的规则列表多于1,则递归可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
if __name__ == "__main__":
    '''
    #关联规则
    dataSet = loadDataSet()
    print(dataSet)
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    print(C1)
    '''
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L,suppData = apriori(mushDatSet,0.3)
    for item in L[3]:
        if item.intersection('2'):print(item)
     
        
        
        