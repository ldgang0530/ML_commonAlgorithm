#!/usr/bin/env python
#_*_coding:utf-8_*_
'KNN算法实现'

from numpy import *
def loadDataSet():
    dataSet = [[1,1],[2,1],[1,2],[2,2],[-1,-1],[-2,-1],[-1,-2],[-2,-2]]
    yLabel = [1,1,1,1,0,0,0,0]
    return dataSet, yLabel

def distCal(x,trainData,p=1.0): #距离度量
    absList=list(map(abs,mat(trainData)-mat(x)))
    sumCalc = 0
    for val in absList[0].A.tolist()[0]:
        sumCalc += val**p
    #sumCalc = sum([val**p for val in absList[0][0])
    distVal = sumCalc**(1.0/p)
    return distVal
#data，要测试的数据
#trainSample,训练样本
#trainLabel，样本类别
#p，选择距离度量的类别，欧氏距离/曼哈顿距离等等
#K，选择最近的k个点进行判断
def kNNClassify(data, trainSample,trainLabel, p=1.0, k=3):
    #最简单实现，计算每一个未知数据与样本数据的距离，然后判断，样本大时不可行
    m,n = shape(trainSample)
    distVec = [distCal(data, trainData, p) for trainData in trainSample]
    distLabel = {}
    for i in range(m):
        distLabel[distVec[i]] = trainLabel[i] #存储距离对应的样本的类别
    distVec = sort(distVec) #对距离进行排序
    countLabel = {} #用于存储k个样本不同类别的个数
    for i in range(k): #遍历距离最小的几个点
        dist = distVec[i]
        if distLabel[dist] not in countLabel: #若没有该类别，就添加
            countLabel[distLabel[dist]] = 0
        countLabel[distLabel[dist]] += 1 #对应label加1
    result = max(countLabel,key=countLabel.get) #获取个数最多的样本类别
    return result