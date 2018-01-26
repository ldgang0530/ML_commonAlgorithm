#!/usr/bin/env python
#_*_coding:utf-8_*_
from numpy import *

def loadDataSet():
    dataSet = [[3,3],[4,3],[1 ,1]]
    yLabel = [1,1,-1]
    return dataSet,yLabel

def fun(x):
    if x>0:
        return 1
    else:
        return -1

def perceptronTrain_1(dataSet, yClass): #实现,1，依次迭代，如果遇到错分类就更新，然后从头开始
    m,n = shape(dataSet)
    print(m,n)
    dataMat = mat(dataSet).T
    yLabel = mat(yClass)

    w = mat(zeros((1,n)))
    b = 0.0
    eta = 0.5
    finished = False
    while not finished:
        count = 0
        for i in range(m):
            yVal = w * dataMat[:,i]+b
            if(yLabel[0,i]*yVal<= 0):
                count += 1
                w += eta*yLabel[0,i]*dataMat[:,i].T
                b += eta*yLabel[0,i]
                break
        if  count == 0:
            finished = True
    return w,b

def perceptronTrain_2(dataSet, yClass): #实现,2，每次更新都遍历数据集，找到偏差最大的数据进行更新
    m,n = shape(dataSet)
    print(m,n)
    dataMat = mat(dataSet).T
    yLabel = mat(yClass)

    w = mat(zeros((1,n)))
    b = 0.0
    eta = 0.5
    finished = False
    while not finished:
        errFlag = {}
        for i in range(m):
            yVal = w * dataMat[:,i]+b
            if yLabel[0,i]*yVal <= 0:
                errFlag[i] = abs(yVal)
        if errFlag:
            maxIndex = max(errFlag, key=errFlag.get)
            w += eta * yLabel[0, maxIndex] * dataMat[:, maxIndex].T
            b += eta * yLabel[0, maxIndex]
        else:
            finished = True
    return w,b

#感知机对偶形式
def perceptronDuality(dataSet, yClass):
    m,n = shape(dataSet)
    dataMat = mat(dataSet).T
    yLabel = mat(yClass)

    alpha = mat(zeros((1,m)))
    b = 0.0
    eta = 0.5
    gram = matmul(dataMat.T,dataMat)
    finished = False
    while not finished:
        count = 0
        for i in range(m):
            yVal = 0
            for j in range(m):
                yVal += alpha[0,j]*yLabel[0,j]*gram[j,i]
            yVal += b
            if(yLabel[0,i]*yVal<=0):
                alpha[0,i] += eta
                b += eta*yLabel[0,i]
                count += 1
        if count == 0:
            finished = True
    w = mat(zeros((n,1)))
    for i in range(m):
        w += alpha[0,i]*yLabel[0,i]*dataMat[:,i]
    return w.T,b

def perceptronClassify(data,w,b):
    n = shape(w)[1]
    dataMat = mat(data).T
    print(w)
    print(b)
    if n != shape(dataMat)[0]:
        raise "wrong data, w and data is not match"
    return fun(w*dataMat+b)


