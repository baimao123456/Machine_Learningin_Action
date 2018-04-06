# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:26:27 2018

@author: baimao
"""
from numpy import *

def loadDataSet(fileName):      #读取以tab键分割的数据
    dataMat = []                #assume last column is target value
    fr = open(fileName)         #打开文件
    for line in fr.readlines(): #读取每一行
        curLine = line.strip().split('\t')  
#        fltLine = map(float,curLine) 这里不能用map，，Python2X返回的是list，，3.X返回的是map对象
        fltLine = [float(item) for item in curLine]# 将每行的内容保存成一组浮点数
        dataMat.append(fltLine)
    return dataMat
#二分数据，dataset中第fature个特征，按value为阈值划分整个数据集，一分为二
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]  #返回大于value的
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:] #返回小于等于value的
    return mat0,mat1
#负责生成叶节点，（返回目标变量的均值）
def regLeaf(dataSet):#返回用于创建叶节点的value，，创建叶子节点时，数据集只有1列
    return mean(dataSet[:,-1])
#计算总的方差，，，直接用均方差乘以样本的个数，，就是总方差--用来描述数据集的混乱度
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]  #用来控制程序的退出
    #如果所有的目标变量相同，则不用再划分， 退出并返回value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet) #叶子节点
    m,n = shape(dataSet)  #m为样本的个数，，n为特征的个数
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)  #errType用来传递函数，，这里S为计算出来的总方差
    bestS = inf; bestIndex = 0; bestValue = 0#初始化bestS为无穷，
    for featIndex in range(n-1):  #！！！对于每个特征 一共2个,,应该是n，，不应该是n-1，，如果是n-1的话，只会取到1个特征
#        print(featIndex)
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):  #对于每个特征的属性值，颜色（红，绿，蓝）
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  #按照spliVal划分特征featIndex
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue   #如果子集的大小小于tolN，则进行根据下一个属性划分
            newS = errType(mat0) + errType(mat1)  #计算划分子集之后的总方差
            if newS < bestS:   #更新最小方差
                bestIndex = featIndex  #更新最佳划分特征
                bestValue = splitVal   #更新最佳特征的阈值
                bestS = newS
#    如果划分数据集后，效果提升的不够明显（方差下降的小于tols），不划分
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #构造叶子节点
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  #找到一个最佳的划分节点
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #如果子集的大小小于tolN，，则停止划分，构造叶子节点
        return None, leafType(dataSet)
    return bestIndex,bestValue#返回最佳的划分特征，，（用索引表示）
                              #返回用于划分的阈值
#创建树的函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#选择最佳划分
    if feat == None: return val #如果满足停止条件，则返回value（回归树是常数，模型树是一个方程）
    retTree = {}  #构造一个字典存放树
    retTree['spInd'] = feat  #分割的特征，构造主节点
    retTree['spVal'] = val   #用来分割的值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)  #根据最佳划分节点划分数据集
    retTree['left'] = createTree(lSet, leafType, errType, ops)  #构建左树，判断条件并递归
    retTree['right'] = createTree(rSet, leafType, errType, ops) #构建右树，判断条件并递归
#    retTree['left'] = createTree(lSet, leafType, errType, ops)  #构建左树，判断条件并递归
    return retTree 

#判断是不是树，如果节点是字典，说明是树----用于判断当前处理的是否是叶子节点还是
def isTree(obj):
    return (type(obj).__name__=='dict')
#如果找到两个叶子节点，，返回树的平均值（对树进行塌陷处理），，也是递归函数
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
#剪枝
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #如果没有足够的test数据进行剪枝，，则返回
    if (isTree(tree['right']) or isTree(tree['left'])):#如果左或者右节点是树，则根据根据树的分割信息，，划分测试数据集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)  #如果左节点是子树，则利用利用之前分割的测试集进行剪枝
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet) #如果右节点是子树，则利用利用之前分割的测试集进行剪枝
    #如果遇到左右节点为叶子节点，，则尝试合并他们，，看看平方是否减少；；如果不是叶子节点，则返回原来的树
    if not isTree(tree['left']) and not isTree(tree['right']):  
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) #按当前特征和属性值划分数据集，
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))  #计算没有合并之前的平方误差
        treeMean = (tree['left']+tree['right'])/2.0               #计算合并节点之后的阈值value，注意如果遇到叶子节点，，则tree['right']是一个数，而不是一棵树
        errorMerge = sum(power(testData[:,-1] - treeMean,2))      #计算合并之后的平方误差
        if errorMerge < errorNoMerge:     #如果平方误差减小了，，则返回合并后的节点value，反之，，返回原来的树，什么操作也没做
            print ("merging")
            return treeMean
        else: return tree
    else: return tree
#用于生成线性模型
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)  #获得数据集和特征的总数
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#创建两个元素全部为1的矩阵
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#获得数据
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)  #计算回归系数
    return ws,X,Y
#创建一个叶子几点的线性模型，并返回系数
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws
#模型的总方差，，
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

#回归树节点预测函数，，model为训练的树，，indata是用来预测所要输入的数据
def regTreeEval(model, inDat):
    return float(model)  #因为回归树的叶子节点是一个常数，，故直接返回model
#模型树节点预测函数
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]      #获得idata的特征个数
    X = mat(ones((1,n+1)))   #构建大小为n+1列的矩阵
    X[:,1:n+1]=inDat         #在原矩阵上增加第0列
    return float(X*model)    #这里的model为线性模型，故可以和x相乘
#获得预测值
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)   #如果不是树，则直接返回节点的值（或者是线性计算出来的值）
    if inData[tree['spInd']] > tree['spVal']:   #如果输入数据idata的某个特征值大于该节点的阈值，则继续往左树判断，，若是小则去右树
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
#获得输入数据的每个特征通过模型获得的预测值 
#        tree为训练出来的树（回归树或者是模型树）
#        testdata为测试需要的数据
#        modelEval  为模型的选择，控制选择是回归树还是模型树
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)         #获得测试集的样本个数，，和预测值的个数相同
    yHat = mat(zeros((m,1)))#构造一个yhat用来存放预测值
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
if __name__ == "__main__": 
#   回归树和模型树比较
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
#    myTree = createTree(testMat,ops=(1,20))
    myTree = createTree(testMat,modelLeaf,modelErr,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
    print(corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
    
    ws,x,y = linearSolve(trainMat)
    print(ws)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
    print(corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
#    print((my2Dat)[:,-1])
    '''
    构建回归树并进行剪枝
    my2Dat = loadDataSet('ex2.txt')
    my2Dat = mat(my2Dat)
    myTree = createTree(my2Dat,ops=(0,1))
    print(myTree)

    myDat2Test = loadDataSet('ex2test.txt')
    myDat2Test = mat(myDat2Test)
    print(prune(myTree,myDat2Test))
    '''
