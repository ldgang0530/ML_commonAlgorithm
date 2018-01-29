#!/usr/bin/env python
#_*_coding:utf-8_*_

import decisionTree


if __name__ == "__main__":
    #dataSet中前两列是特征值，最后一列是类别标签
    dataSet = [['a', 'b', 'yes'], ['a', 'b', 'yes'], ['a', '-b', 'no'], ['-a', 'b', 'no'], ['-a', 'b', 'no']]
    labels = ['aLabel', 'bLabel'] #对应各特征的意义，这里只有两列
    tree = decisionTree.constructTree(dataSet,labels)
    print(tree)
    result = decisionTree.treeClassify(['a','b'],tree, labels)
    print(result)