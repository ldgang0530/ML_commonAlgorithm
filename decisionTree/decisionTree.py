#!/usr/bin/env python
#_*_coding:utf-8_*_
'介绍了决策树实现分类， ID3/C4.5实现基本的决策树，以及CART'
from numpy import *
'''
首先，确定最佳的特征值（计算熵，比较信息增益，选择最优分类特征值作为最佳的特征值）
然后，按照选定的特征值，分割
最后，对子集重复步骤1,2。退出条件：没有特征值可以再分；增益已基本没有提升
最优判定：
ID2--信息增益最大
C4.5--增益比最大
'''

def calShannon(dataSet):
    dataMat = mat(dataSet)
    yLabel = dataMat[:,-1] #最后一列是类别
    numLabel = len(dataSet)
    countLabel = {}
    for perLabel in yLabel: #统计每一个类别的个数
        if perLabel[0,0] not in countLabel:
            countLabel[perLabel[0,0]] = 0
        countLabel[perLabel[0,0]] += 1
    shonn = 0.0
    for i in countLabel.keys(): #计算熵
        prob = float(countLabel[i])/numLabel
        shonn += -prob * log(prob)
    return shonn

def selectBestFeature_ID3(dataSet): #选择最佳的，ID3策略，选择信息增益最大的列
    m,n = shape(dataSet)
    dataMat = mat(dataSet)
    yLabel = dataMat[:,-1]
    minShonn = float('inf')
    bestIndex = -1
    if ( not m ) or ( not n ):
        return -1
    for i in range(n-1): #遍历特征值，最后一列是类别，故减去1
        subShonn = 0.0
        subSet = {}
        for j in range(m): #遍历每一条记录
            if dataMat[j,i] not in subSet: #按照第i列的特征值将数据集分成n个子集
                subSet[dataMat[j,i]] = []
            subSet[dataMat[j,i]].append(dataSet[j][:i]+dataSet[j][i+1:])
        for k in subSet.keys():
            tmpSet = subSet[k]
            subProb = float(len(tmpSet))/m
            subShonn += subProb*calShannon(tmpSet) #计算子集的条件熵，求和
        if minShonn > subShonn: #这里是ID3，信息增益baseShonn-subShonn最大，也就是subShonn最小
            minShonn = subShonn
            bestIndex = i
    return bestIndex

def selectBestFeature_C45(dataSet): #选择最佳的，C4.5策略，选择信息增益比最大的策略
    m,n = shape(dataSet)
    baseShonn = calShannon(dataSet)
    dataMat = mat(dataSet)
    yLabel = dataMat[:,-1]
    maxGainProportion = float('-inf')
    bestIndex = -1
    if ( not m ) or ( not n ):
        return -1
    for i in range(n-1): #遍历特征值，最后一列是类别，故减去1
        subShonn = 0.0
        subSet = {}
        for j in range(m): #遍历每一条记录
            if dataMat[j,i] not in subSet: #按照第i列的特征值将数据集分成n个子集
                subSet[dataMat[j,i]] = []
            subSet[dataMat[j,i]].append(dataSet[j][:i]+dataSet[j][i+1:])
        for k in subSet.keys():
            tmpSet = subSet[k]
            subProb = float(len(tmpSet))/m
            subShonn += subProb*calShannon(tmpSet) #计算子集的条件熵，求和
        subGainProportion = (baseShonn - subShonn)/baseShonn
        if maxGainProportion < subGainProportion: #这里是C4.5，信息增益比(baseShonn-subShonn/baseShonn)最大，也就是subShonn最小
            maxGainPropotion = subGainProportion
            bestIndex = i
    return bestIndex

def splitDataSet(dataSet, index):
    countLabel = {}  # 记录每一个类别的子集，以index列的值为键，以数据列表为集合
    m = len(dataSet)
    for i in range(m):
        if dataSet[i][index] not in countLabel:
            countLabel[dataSet[i][index]] = []
        countLabel[dataSet[i][index]].append(dataSet[i][:index]+dataSet[i][index+1:])
    return countLabel

def majorityCnt(dataSet): #统计多数类别，缤纷返回,也可把这个设置为其他标识，比如不确定
    classCount = {}
    for vote in dataSet:
        if vote[-1] not in classCount.keys():classCount[vote[-1]]=0
        classCount[vote[-1]]+=1
    sortedClassCount = sorted(classCount.items(),\
        key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def constructTree(dataSet,labels):
    def travel(dataSet,labels):
        result = {}
        yLabelMat = mat(dataSet)[:,-1]
        yLabel = []
        for val in yLabelMat:
            yLabel.append(val[0,0])
        print(yLabel,type(yLabel))
        if  len(set(yLabel)) == 1: #若只有一类，则不再递归分割
            result = yLabel[0]
            return result
        elif len(dataSet[0]) == 1: #没有属性可分
            return majorityCnt(dataSet)
        bestIndex = selectBestFeature_ID3(dataSet)  #ID3
        #bestIndex = selectBestFeature_C45(dataSet) #C4.5
        subDict = splitDataSet(dataSet,bestIndex)
        subLabel = labels[:bestIndex]+labels[bestIndex+1:]
        tmpVec = {}
        for i in subDict:
            if(subDict[i]):
                tmpResult = travel(subDict[i],subLabel)
                tmpVec[i]=tmpResult  #i表示字典的key
        result[labels[bestIndex]] = tmpVec
        return result
    return travel(dataSet,labels)

def treeClassify(testData,tree, colLabels):
    firstStrSides = list(tree.keys())  # 获取标签字符串
    firstStr = firstStrSides[0]
    secondDict = tree[firstStr]  # 标签对应的子数据集
    featIndex = colLabels.index(firstStr)  # 将标签字符串转换为索引
    for key in secondDict.keys():  # 遍历整棵树
        if testData[featIndex] == key:  # 比较testVec中值与树节点的值，若达到叶子节点，则返回当前节点的分类标签
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = treeClassify(testData,secondDict[key], colLabels)
            else:
                classLabel = secondDict[key]
    return classLabel