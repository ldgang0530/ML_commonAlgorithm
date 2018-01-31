#!/usr/bin/env python
#_*_coding:utf-8_*_
import adaBoost

if __name__ == "__main__":
    dataSet,classLabel = adaBoost.loadDataSet()
    classifyVec = adaBoost.adaBoostTrain(dataSet,classLabel,5)
    print(classifyVec)

    testData = [[1.5,2.0],[1.0,1.5]]
    result = adaBoost.adaBoostTest(testData,classifyVec)
    print(result)



