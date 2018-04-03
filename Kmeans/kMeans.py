# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:34:05 2018

@author: baimao
"""

from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(fileName):      #读取以tab作为分割的float数据
    dataMat = []                #最后一列是标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
#        fltLine = map(float,curLine) #map all elements to float()
        fltLine = [float(item) for item in curLine]
        dataMat.append(fltLine)
    return dataMat
#计算两个向量之间的距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
#构建一个包含k个随机质心的集合，，质心必须在数据集的范围内，，k为簇心的个数
def randCent(dataSet, k):
    n = shape(dataSet)[1]  #获取数据的维度，这里为二维的
    centroids = mat(zeros((k,n)))   #建立簇心矩阵
    for j in range(n):   #按每个维度建立随机簇中心, 在每个维度范围内
        minJ = min(dataSet[:,j]) #得到每个维度的最小值     
        rangeJ = float(max(dataSet[:,j]) - minJ)   #最大最小的范围
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #random.rand(m,n)会 返回一个m*n的小于1的随机数
#        print(centroids[:,j])
    return centroids
#kmeans算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  #获得样本的个数，，在2D图中就是点的个数
    clusterAssment = mat(zeros((m,2)))#簇分配结果矩阵，第一列 纪录簇的索引值，第二列存储当前点到簇质心的距离
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True                     #簇心改变标志位，初始化为改变了
    while clusterChanged:                     #如果簇心改变了
        clusterChanged = False
        for i in range(m):                    #将每个点分配到最近的质心
            minDist = inf; minIndex = -1      #minDist为最短距离
            for j in range(k):                #找到距离点i最近的簇点
                distJI = distMeas(centroids[j,:],dataSet[i,:])  #计算每个点到各个簇心的距离
                if distJI < minDist:                            #找到距离最近的簇心
                    minDist = distJI; minIndex = j              #找到最近距离，并将最近簇心的索引给minIndex
            if clusterAssment[i,0] != minIndex: clusterChanged = True  #如果当前的分配结果和之前的不一样，，说明簇心改变了
            clusterAssment[i,:] = minIndex, minDist**2       #更新簇心和平方差
#        print (centroids)
        for cent in range(k):#根据划分的点的平均值重新计算簇点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]  #得到某个簇心上的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0)  #求这些点的平均值，最为新的簇心
#            plt.scatter(x = ptsInClust[:,0].T.tolist()[0],y = ptsInClust[:,-1].T.tolist()[0])#画图
#    如果簇心不再改变，，则退出
    return centroids, clusterAssment#返回簇心和各点的簇心分配结果
#画出每个簇心
def plot(datMat,centroids,clusterAssment):
    plt.scatter(x = centroids[:,0].T.tolist()[0],y = centroids[:,-1].T.tolist()[0],marker = '+',c = 'r')
#二分K-均值算法,,!!一直将簇0，作为要分割的簇
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]    #获得样本的个数，，在2D图中就是点的个数
    clusterAssment = mat(zeros((m,2)))    #簇分配结果矩阵,clusterAssment初始化所有的数据属于同一个簇-- 簇0 
    centroid0 = mean(dataSet, axis=0).tolist()[0]  #计算所有数据的各个维度的平均值（将这个平均值作为初始化的簇心或者叫中心）
    centList =[centroid0] #同来存储簇点，，len(cenList)为簇点的个数，，第一个簇点是总数据的平均值
    for j in range(m):    #对于每个点计算该点到每个簇心的距离(误差)，，取平方是为了重视远离中心的点
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2   #
    while (len(centList) <= k):  #当前簇的数目小于k时,
        lowestSSE = inf
        for i in range(len(centList)):   #依次对每个簇进行划分 ，划分的标准是能降低SSE的值
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]      #得到属于簇心i的所有数据
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)      #对属于簇心i的数据，进行二划分
#            print('splitClustAss',splitClustAss)
            plt.scatter(x = ptsInCurrCluster[:,0].T.tolist()[0],y = ptsInCurrCluster[:,-1].T.tolist()[0])
            sseSplit = sum(splitClustAss[:,1])      #存储进行划分之后的数据的平方误差SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])   #存储没有进行划分的数据的平方误差SSE,,.A是将matrix作为array返回
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:       #如果划分之后误差减小了，更新最佳划分簇心
                bestCentToSplit = i                        #i为簇心的索引，将
                bestNewCents = centroidMat                 #更新最佳簇心
                bestClustAss = splitClustAss.copy()        #更新最佳簇心分配结果
                lowestSSE = sseSplit + sseNotSplit         #更新最小平方误差
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #  #数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit  #将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号连续不出现空缺
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#更新最佳划分簇点bestCentToSplit
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#更新每个点的最好的分配结果
    return mat(centList), clusterAssment
def load_place_info(filename):  #读取经纬度信息
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])  
    datMat = mat(datList)
    print(datMat)
def distSLC(vecA, vecB):#球面余弦定理计算两点之间的距离
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy
import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')    #基于图片建立矩阵
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()  
if __name__ == "__main__":
    datMat = loadDataSet('testSet2.txt')
    datMat = mat(datMat)
    '''
    datMat = mat(datMat)
    centroids, clusterAssment = (kMeans(datMat,4))
    plot(datMat,centroids,clusterAssment)
    '''
    clusterClubs(3)