# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:34:50 2018
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  
@author: baimao
"""
#树中节点的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  #名字
        self.count = numOccur #出现的次数
        self.nodeLink = None  #用于连接相似的元素项
        self.parent = parentNode      #指向当前节点的父节点
        self.children = {}            #dict变量，用来存放子节点
    
    def inc(self, numOccur):  #增加元素出现的次数
        self.count += numOccur
        
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)#打印名字和出现的次数，，'  '*ind是打印ind个空格
        for child in self.children.values():  #递归遍历树节点
            child.disp(ind+1)
def createTree(dataSet, minSup=1): #从数据集中创建 FP-tree  but don't mine
    headerTable = {}               #头指针列表，用来指向给定类型的第一个实例，除了存放指针，还可用来保存每类元素的总数
    #遍历两次数据集
    for trans in dataSet:#第一次遍历获得每个元素项的出现频率
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
                               #获得原来item的值，然后加1
#    for k in headerTable.keys():  #去除不满足minSup条件的元素（出现次数小于 minSup）
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup: 
            del(headerTable[k])  #删除不满足条件的元素
    freqItemSet = set(headerTable.keys())
    #print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #如果没有满足minsupport的元素，返回空
    for k in headerTable:  #增加指针项，指向该类型的第一个实例
        headerTable[k] = [headerTable[k], None] #用Nodelink ，来填充headtable类似headerTable:  {'z': [5, None], 'r': [3, None]}
    retTree = treeNode('Null Set', 1, None) #建立树
    for tranSet, count in dataSet.items():  #第二次遍历数据集
        localD = {}
        for item in tranSet:  #根据全局频率对每个事务中的元素进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #['z', 'x', 'y', 's', 't'] orderedItems,按出现频率排序
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table
#更新树，使树增长
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#判断元素（orderedItems[0]）是否在retTree的子节点，如果在，直接对这个节点的出现次数加1
        inTree.children[items[0]].inc(count) #增加出现次数
    else:   #如果树的节点中没有这个item，则新建一个节点，然后加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #更新header指针表 ,,如果headerTable中对应的item的指针为NOne，则将当前的节点给这个指针
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() 更新剩下的item
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        # updateTree()完成的最后一件事是不断迭代调用自身，每次调用时会去掉列表中的第一个元素
#更新headerTable，确保节点链接指向树中的每一个实例
def updateHeader(nodeToTest, targetNode):   #这里没有用到递归
    while (nodeToTest.nodeLink != None):   #判断是否到了子节点 #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode  #如果到了子节点，，则将原来的子节点会指向一个新得节点
#创建一个数据集
def loadSimpDat():
    simpDat = [['z', 'r', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
#创建初始集合
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
#从子节点到根节点，遍历树的节点来找到以leafNode为结尾的前缀路径
def ascendTree(leafNode, prefixPath): #
    if leafNode.parent != None:  #如果没有到达根节点，将当前子节点的名字保存到prefixPath
        prefixPath.append(leafNode.name)  
        ascendTree(leafNode.parent, prefixPath)  #继续迭代寻找下一节点
#遍历链表直到到达结尾  
def findPrefixPath(basePat, treeNode): #treeNode来自于headerTable中的指针
    condPats = {}
    while treeNode != None: #用来遍历所有实例
        prefixPath = []
        ascendTree(treeNode, prefixPath)  #寻找该实例的前缀路径
        if len(prefixPath) > 1:   #判断路径的条数,显示路径大于1的实例点
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink  #遍历完同类的一个第一个实例，接着寻找下一个实例（属于同一类比如都是'r'或者'z'）
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
#    print(headerTable.items())
    # 对头指针表中元素项按照其出现频率进行排序，默认是从小到大
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
#        print('finalFrequent Item: ',newFreqSet)    #append to set
        freqItemList.append(newFreqSet) # 将每个频繁项添加到频繁项集列表freqItemList中
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])# 使用findPrefixPath()创建条件基
#        print('condPattBases :',basePat, condPattBases)
        # 将条件基condPattBases作为新数据集传递给createTree()函数
        # 这里为函数createTree()添加足够的灵活性，确保它可以被重用于构建条件树
        myCondTree, myHead = createTree(condPattBases, minSup)
#        print('head from conditional tree: ', myHead)
        # 如果树中有元素项的话，递归调用mineTree()函数
        if myHead != None: #3. mine cond. FP-tree
#            print('conditional tree for: ',newFreqSet)
#            myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
if __name__ == "__main__":
    '''
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree,myHeaderTab = createTree(initSet,3)
#    print(findPrefixPath('z',myHeaderTab['z'][1]))  #myHeaderTab['r'][1]为指向r的第一个实例
    freqItems = []
    mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
    print(freqItems)
    '''
    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree,myHeaderTab = createTree(initSet,3)  
    freqItems = []
    mineTree(myFPtree,myHeaderTab,100000,set([]),freqItems)
    print(freqItems)
    
    
    