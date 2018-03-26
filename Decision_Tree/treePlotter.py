# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:37:21 2018

@author: baimao
"""

from matplotlib import pyplot as plt 

decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")#定义文本框和箭头格式
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
def plotNode(nodeTxt, centerPt, parentPt, nodeType):  #绘制带箭头的注解
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
'''#旧版本
def createPlot():
    fig = plt.figure(1,facecolor ='white')
    fig.clf()  
    createPlot.ax1 = plt.subplot(111,frameon = False)
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
    '''
def createPlot(inTree):#新版本画树的函数
    fig = plt.figure(1, facecolor='white')  #设置面板的颜色
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))#获得节点的总数
    plotTree.totalD = float(getTreeDepth(inTree))  #获得树的深度
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
def getNumLeafs(myTree):   #获得节点的个数
    numLeafs = 0
    firstStr = list(myTree.keys())[0]  #获得第一个节点的名字,,这里和例子不一样，注意！！！
    secondDict = myTree[firstStr]  #获得第一个节点下边的树结构
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':#判断此节点（dict类型）下面是否还有其他子节点，如果有继续迭代，，
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1   #如果没有，直接加1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]  #获得第一个节点的名字
    secondDict = myTree[firstStr]  #获得第一个节点下边的树结构
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':#判断此节点（dict类型）下面是否还有其他节点，如果有继续迭代，，
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1   #如果没有,深度为1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth    #返回最大深度
def retrieveTree(i):  #构造一棵树
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]    
def plotMidText(cntrPt,parentPt,txtString):#在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]   # x轴
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]   # y轴
    createPlot.ax1.text(xMid,yMid,txtString)
#  plotTree.xOff和plotTree.yOff 用来追踪已经绘制的节点位置，以及放置下一节点的适当位置
#  totalW，totalD记录树的宽度和深度
def plotTree(myTree, parentPt, nodeTxt):#画出树，，递归函数
    numLeafs = getNumLeafs(myTree)  #得到树在x轴上的宽度
    depth = getTreeDepth(myTree)    #得到树在y轴上的深度
    firstStr = list(myTree.keys())[0]  #获得第一个节点的名字
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)#计算子节点的坐标
    plotMidText(cntrPt, parentPt, nodeTxt)   #在父子节点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)   #画出父节点
    secondDict = myTree[firstStr]     #获得第二个节点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD    #按比例减少全局变量plotTree.yOff（自顶向下画出）   
    for key in secondDict.keys():   #画出子节点，如果下边还有其他节点，，继续迭代画出
        if type(secondDict[key]).__name__=='dict':#判断节点是不是dict类型, 如果不是，则为叶子节点   
            plotTree(secondDict[key],cntrPt,str(key))        #如果是dict类型，说明下边还有子节点，打印子节点下面的父节点
        else:   #是叶子节点，，并打印出来
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  #计算叶子节点的坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)#画出叶子节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  #画出标注文本信息
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD    #！！！很重要，，画完父节点，，坐标要往上走走，，来画其他的节点 
