#!/usr/bin/env python
#_*_coding:utf-8_*_
'KD树实现kNN算法'

from numpy import sqrt

#k:特征值数目，k维
#split：当前节点在第几维做的分割
#left：左子树的节点
#right：右子树的节点
class KDNode():
    def __init__(self, nodeData,split, left, right):
        self.nodeData = nodeData;
        self.split = split  #进行分割的维度序号
        self.left = left
        self.right = right

class KDTree():
    def __init__(self,dataSet,yLabel):
        self.k = len(dataSet[0]) #数据的维度,这里虚线统计，因为下面就修改了样本集，把类别添加到了最后
        for i in range(len(dataSet)): #将类别插入到最后一列，这里需要遍历样本集
            dataSet[i].append(yLabel[i])
        self.root = self.createTree(0,dataSet)
    def createTree(self,split,dataSet):
        if not dataSet: #若已分割完，退出递归
            return None
        dataSet.sort(key=lambda X: X[split]) #以第split维的数值排序
        splitNodeIndex = len(dataSet)//2  #取中位数
        midData = dataSet[splitNodeIndex]
        splitNext = (split+1) % self.k
        return KDNode(midData,split,
                     self.createTree(splitNext,dataSet[:splitNodeIndex]),
                     self.createTree(splitNext,dataSet[splitNodeIndex+1:]))
    def getOrder(self):
        def preorder(root): #先序遍历
            if not root:
                return
            else:
                print(root.nodeData)
                preorder(root.left)
                preorder(root.right)
        preorder(self.root)
class resultNodes:
    def __init__(self,nodeData, maxDist):
        self.nodeData = nodeData
        self.maxDist = maxDist
def find_nearest(testData,tree):
    k = len(testData)
    def travel(kdNode,target, maxDist):
        if kdNode is None: #若kdNode为空，则返回最大距离
            return resultNodes([0]*k, float('inf'))
        s = kdNode.split #在第split维进行的分割
        pivot = kdNode.nodeData #节点的样本
        if target[s] <= pivot[s]: #如果小于样本，到左子树，同时记录右子树的信息
            nearerNode = kdNode.left
            furtherNode = kdNode.right
        else: #否则，进入右子树，记录左子树的信息
            nearerNode = kdNode.right
            furtherNode = kdNode.left
        temp1 = travel(nearerNode,target, maxDist) #遍历查找包含目标点的区域
        nearest = temp1.nodeData #更新最近点
        dist = temp1.maxDist #最近点与目标点的距离
        if dist < maxDist:
            maxDist = dist
        tempDist = abs(pivot[s]-target[s]) #以目标点为球心，maxDist为半径的超球体
        if maxDist < tempDist: #如果分割点(父节点)与超球体不相交
            return resultNodes(nearest, dist) #直接返回
        tempDist = sqrt(sum((p1-p2)**2 for p1,p2 in zip(pivot[:-1], target[:-1]))) #计算距离
        if tempDist < dist: #如果targe与父节点的距离 比 目前的dist更小
            nearest = pivot #更新最近点
            dist = tempDist #更新最近距离
            max_dist = dist #更新超球体半径
        temp2 = travel(furtherNode,target,maxDist) #检查另一个子树对应的区域是否有更近的点
        if temp2.maxDist < dist: #若存在
            nearest = temp2.nodeData  #更新最近点
            dist = temp2.maxDist #更新最近距离
        return resultNodes(nearest,dist)
    return travel(tree.root,testData,float("inf")) #从根节点递归

#查找最近的n个点，返回list，每一个元素的最后一个值是类别，其余值是特征值
def find_nearest_n(testData,tree,n):#查找最邻近的n个点
    k = len(testData)
    result = [] #记录结果node
    maxVal = [] #记录node对应的与目标点的距离
    def getDist(point1,point2): #计算距离，欧式距离
        return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1[:-1], point2[:-1])))
    def appendResult(node, dist): #向结果列表中添加node
        if (len(result) < n):
            result.append(node)
            maxVal.append(dist)
        elif dist<max(maxVal):
            maxIndex = maxVal.index(max(maxVal))
            del maxVal[maxIndex]
            del result[maxIndex]
            result.append(node)
            maxVal.append(dist)

    def travel(kdNode,target): #递归查找
        if kdNode is None: #若kdNode为空，则返回
            return
        nodeDist = getDist(kdNode.nodeData,target) #计算node与目标点的距离
        appendResult(kdNode.nodeData, nodeDist) #判断，并添加node
        s = kdNode.split #在第split维进行的分割
        pivot = kdNode.nodeData #节点的样本
        splitPlane = pivot[s]  #分类超平面
        planeDist = target[s] - splitPlane #目标点与分类超平面的距离
        planeDist2 = planeDist**2 #平方

        if target[s] <= pivot[s]: #如果小于样本，到左子树，同时记录右子树的信息
            nearerNode = kdNode.left
            furtherNode = kdNode.right
        else: #否则，进入右子树，记录左子树的信息
            nearerNode = kdNode.right
            furtherNode = kdNode.left
        travel(nearerNode,target) #进一步遍历查找包含目标点的区域
        if planeDist2 < max(maxVal) or len(result) <n: #目标点与超平面的距离小于当前结果列表中最大的距离或者结果node数目不够n个
            travel(furtherNode, target)  # 遍历另一个子节点
    travel(tree.root,testData) #从根节点递归
    return result #find_nearest_n的结果返回

def KDTreeClassify(testData,dataSet,yLabel, k):
    kd = KDTree(dataSet,yLabel)
    result_K = find_nearest_n(testData, kd, k)  # 最近的k个点
    countLabel = {}
    for record in result_K:
        recordLabel = record[-1]
        if recordLabel not in countLabel:
            countLabel[recordLabel] = 0
        countLabel[recordLabel] += 1
    return max(countLabel, key=countLabel.get) #最近k个点中对应最多的类别作为结果
if __name__ == "__main__":
    data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    yLabel = [0,1,0,1,0,1]
    testData = [9,6]
    #kd = KDTree(data) ，经过这建树，data结构已经修改，不能再运行KDTreeClassify，故不可与下面的共存
    #kd.getOrder()
    #ret = find_nearest(testData,kd) #最近邻点
    #print(ret.nodeData)
    result = KDTreeClassify(testData,data, yLabel,3) #可直接调用，内含了kd树的构建，返回最近的k个点
    print(result)