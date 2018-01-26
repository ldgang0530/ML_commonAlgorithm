#!/usr/bin/env python
#_*_coding:utf-8_*_

#感知机算法，调用示例。数据集在loadDataSet中设定了，也可以从文件读取

import perceptron

if __name__ == "__main__":
    print("感知机程序：")
    data = input()
    trainData,yLabel = perceptron.loadDataSet()
    w,b = perceptron.perceptronTrain_1(trainData,yLabel)
    print("train1  result:",perceptron.perceptronClassify(data,w,b))
    w,b = perceptron.perceptronTrain_2(trainData,yLabel)
    print("train2  result:",perceptron.perceptronClassify(data,w,b))
    w, b = perceptron.perceptronDuality(trainData, yLabel)
    print("result:", perceptron.perceptronClassify(data, w, b))