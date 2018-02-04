#!/usr/bin/env python
#_*_coding:utf-8_*_
from numpy import *

def loadDataSet():
    dataSet = [[1,1],[1,0],[1,0],[1,0],[1,0],[2,0],[2,1],[2,1],[2,2],[2,2],[3,2],[3,1],[3,1],[3,2],[3,2]]
    yLabel = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    return dataSet, yLabel

def train(dataSet,yLabel):
    m,n = shape(dataSet)
    allNum = len(yLabel)

    classProb = {}
    subSet = {}
    for i in range(allNum):
        if yLabel[i] not in classProb:
            classProb[yLabel[i]] = 0
            subSet[yLabel[i]] = []
        classProb[yLabel[i]] += 1
        subSet[yLabel[i]].append(dataSet[i]) #按照类别将数据分隔成子集

    featureCondition = {}  #用于统计不同属性中不同的属性值情况
    for i in range(n):
        featureVal = []
        for j in range(m):
            if dataSet[j][i] not in featureVal:
                featureVal.append(dataSet[j][i])
        featureCondition[i] = featureVal

    K = len(classProb) #有多少类别
    lamda = 1
    for key in classProb:
        classProb[key] = log(1.0*(classProb[key]+lamda)/(allNum+K*lamda))  #每一个类别的概率

    cSubSetFeature = {} #有三层字典，第一级类别，第二级属性特征，第三级属性特征值统计
    for key in subSet:
        cSub = subSet[key] #某一类别的数据集
        m_sub,n_sub = shape(cSub)
        allFeature = {}
        for i in range(n_sub): #访问某一特征
            featureVal = {}  # 存储某一个特征的值的情况
            for j in featureCondition[i]:
                featureVal[j] = 1
            for j in range(m_sub): #遍历数据集
                featureVal[cSub[j][i]] += 1
            allFeature[i] = featureVal
        cSubSetFeature[key] = allFeature  #将某一类别的属性值情况存储

    #统计不同属性值在不同类别下的概率情况
    prob = {}
    for key in cSubSetFeature:
        cSub = cSubSetFeature[key]
        cSubProb = {}
        for i in cSub: # 访问不同属性
            featureSub = cSub[i]
            allStastic = sum([featureSub[j] for j in featureSub])
            featureSubProb = {}
            allLen = len(featureSub)
            for j in featureSub:
                featureSubProb[j] =log(1.0*(featureSub[j]+1)/(allStastic+allLen))
            cSubProb[i] = featureSubProb
        prob[key] = cSubProb
    return  classProb, prob  #classProb，类别的概率； prob记录了不同类别中不同属性对应的概率

def test(testData,classProb,prob):
    p = {}
    for key in classProb:
        cla = prob[key]
        tempProb = sum([cla[i][testData[i]] for i in range(len(testData))])
        p[key] = tempProb+classProb[key]
    return max(p,key=p.get)

if __name__ == "__main__":
    dataSet,yLabel = loadDataSet()
    classProb,prob = train(dataSet,yLabel)  #prob为字典，第一级是类别，第二级是属性，第三级是属性的概率，都是取了对数
    testData = [2,0]
    y = test(testData,classProb,prob)
    print(y)










