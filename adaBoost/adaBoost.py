#!/usr/bin/env python
#_*_coding:utf-8_*_
from numpy import *

def loadDataSet(): #加载数据集
    dataSet = [[1,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]]
    classLabels = [1,1,-1,-1,1]
    return dataSet, classLabels

def baseClassify(dataMat, dim, threshVal, threshIneq): #基本分类器
    m,n = shape(dataMat) #数据集
    retArr = ones((m,1)) #用于记录判别结果，每个记录都有一个结果
    if threshIneq == 'lt' : #小于val设为-1
        retArr[dataMat[:,dim]<=threshVal] = -1
    else:
        retArr[dataMat[:,dim]>threshVal] = -1
    return retArr
#dataSet:数据集
#classLabels:类别标签
#omga：用于传递每个样本的权重，行向量
def buildBaseTree(dataMat, labelMat,omega):  #寻找错误率最低的单层决策树
    m,n = shape(dataMat)
    minErr = float('inf')
    bestStump = {}
    bestClassPredict = mat(zeros((m,1)))  #记录预测结果
    for i in range(n): #遍历所有的特征值
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        steps = 10.0  #寻找决策边界的迭代次数
        stepsVal = (rangeMax - rangeMin)/steps  #决策步进
        for j in range(-1,int(steps)+1):
            for inequal in ['lt','gt']:
                isRight = mat(ones((m, 1)))  # 先认为所有的判定都是错误的，标记为1
                threshVal = rangeMin + j*stepsVal
                predictedVal = baseClassify(dataMat,i,threshVal,inequal)
                isRight[predictedVal==labelMat] = 0  #正确的标记为0
                err = omega*isRight  #计算错误率
                if err < minErr:
                    minErr = err
                    bestClassPredict = predictedVal.copy()
                    bestStump['dim'] = i
                    bestStump['threshIneq'] = inequal
                    bestStump['threshVal'] = threshVal
    return bestStump, minErr, bestClassPredict  #返回决策树的特征选择，equal标识以及决策边界

#dataSet,数据集
#classLabels，类别标签
#iter,迭代次数
def adaBoostTrain(dataSet,classLabels,iter = 5): #自适应提升树
    dataMat = mat(dataSet)
    m = shape(dataMat)[0]
    classMat = mat(classLabels).T
    omega = 1.0*ones((1,m))/m  #共m个数据集，默认每个数据集的权重相同,1/m
    classifyVec = []
    estResult = mat(zeros((m,1)))
    for i in range(iter):
        bestStump,minErr,bestClassPredict = buildBaseTree(dataMat,classMat,omega)  #训练得到最佳的分类器，最小误差，最佳的预测结果
        alpha = float(0.5*log((1.0-minErr)/max(minErr,1e-16))) #记录alpha
        bestStump['alpha'] = alpha
        classifyVec.append(bestStump)
        expon = multiply(-1.0*alpha*classMat.T,bestClassPredict)
        omega = omega*exp(expon)
        omega = omega/omega.sum()
        #错误累加，计算总的错误率，若满足要求，直接退出，不必等待到达迭代次数
        estResult += alpha*bestClassPredict  #多个分类器结果累加
        errRate = 1.0*sum(sign(estResult) != classMat)/m #错误率
        if errRate == 0.0:
            break
    return classifyVec
#testData,测试函数
#classifyVec,弱分类器集合，必须按先后顺序调用
def adaBoostTest(testData, classifyVec): #测试函数
    m = len(testData)
    n = len(classifyVec)
    testDataMat = mat(testData)
    resultLabel = mat(zeros((m,1)))
    for i in range(n):
        dim = classifyVec[i]['dim']
        threshIneq = classifyVec[i]['threshIneq']
        threshVal = classifyVec[i]['threshVal']
        classLabel=baseClassify(testDataMat, dim, threshVal, threshIneq) #利用弱分类器分类，得到分类结果
        resultLabel += classifyVec[i]['alpha']*classLabel #分类结果，alpha加权
    return sign(resultLabel)


