#!/usr/bin/env python
#_*_coding:utf-8_*_

#kNN算法，调用示例。数据集在loadDataSet中设定了，也可以从文件读取
from numpy import mat
import kNN

if __name__ == "__main__":
    print("kNN近邻程序,输入：A B，如 1 2")
    data = mat(input()).tolist()
    trainData,yLabel = kNN.loadDataSet()
    result = kNN.kNNClassify(data, trainData, yLabel, 2,k=3)
    print(result)
