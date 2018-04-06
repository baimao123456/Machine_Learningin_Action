# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:58:14 2018

@author: baimao
"""

from numpy import *
from numpy import linalg as la

def loadExData():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
#计算相似度，欧氏距离   
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))#norm为计算2范数，就是两个向量的距离
#皮尔逊系数
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0 #判断是否存在3个以上的点
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]#计算皮尔逊系数，并将区间放到 0-1之间
#余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)#将区间放到 0-1之间
#基于物品相似度的推荐引擎，
#    standEst函数计算用户user对于物品item的估计评分值
#行对应用户，列对应物品
#输入参数 dataMat为数据，user为用户编号，simMeas为相似度计算方法，item为物品编号
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]  #获得物品的种类数
    simTotal = 0.0; ratSimTotal = 0.0 #总相似度 和 总评分
    for j in range(n):
        userRating = dataMat[user,j]  #判断用户 user是否对 物品j进行了评分，这个过程用来寻找一个用户已经评级的物品
        if userRating == 0: continue  #如果没有，继续循环
        #寻找两个（用户都评级的物品）
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0] 
#        logical_and 为逻辑与
        if len(overLap) == 0: similarity = 0 #如果没有两个（用户都评级的物品）则相似度为0
        else: similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j]) #如果有，则计算物品之间的相似度
        print('the %d and %d similarity is: %f' % (item, j, similarity)) #打印物品item和j的相似度
        simTotal += similarity                 #计算总相似度
        ratSimTotal += similarity * userRating #计算总评分
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal #返回用户对物品item的评分  
#给用户推荐3个物品
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#建立一个用户没有评级的物品列表，用来推荐
    if len(unratedItems) == 0: return 'you rated everything' #如果都已评级，则不用推荐
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))  #对每个未评级的物品进行评分
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]  #按从大到小顺序排列
#基于SVD的评分估计，对数据进行降维后，再进行推荐
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)  #进行奇异值分解
    Sig4 = mat(eye(4)*Sigma[:4]) #构造一个对角矩阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #由主要的维度信息构造新的数据
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1),
            else: print(0),
        print('')
#图片压缩，numSvd控制选取几个维度的信息，，thresh控制判断是还是1的阈值
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)  #根据阈值输出1还是0，这里阈值默认是0.8
    U,Sigma,VT = la.svd(myMat)  #奇异值分解
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
if __name__ == "__main__":
    myData = mat(loadExData2())
#    print(recommend(myData,1,estMethod=standEst),'standEst')
    print(recommend(myData,1,estMethod=svdEst),'svdEst')
    
#    imgCompress()#图片压缩示例
    '''svd  test
    Data = loadExData()
    U, Sigma ,VT = linalg.svd(Data)
    Sig3 = mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
    print(U[:,:3]*Sig3*VT[:3,:])
    '''
   