#!/usr/bin/env python
#_*_coding:utf-8_*_
'''
    对应李航博士《统计学习方法》
    程序参考链接：
    http://blog.csdn.net/wds2006sdo/article/details/75212599
    https://github.com/WenDesi/lihang_book_algorithm/blob/master/hmm/hmm.py
'''
from numpy import *
class HMM:
    def __init__(self,N,M): #N表示可能的状态数，M表示可能的观测数
        self.A = mat(zeros((N,N))) #状态转移概率矩阵
        self.B = mat(zeros((N,M))) #观测概率矩阵
        self.pai = mat(1.0/N*ones((N,1))) #初始概率向量

        self.N = N  #可能的状态数
        self.M = M  #可能的观测数
        self.randomInit()  #随机生成矩阵A和B

    def forward(self): #前向处理
        #公式10.15
        self.alpha[0,:] = multiply(self.pai.T,self.B[:,self.observeQueue[0]].T)
        #公式10.16
        for t in range(1,self.tNum):
            for i in range(self.N):
                sumVal = 0.0
                for j in range(self.N):
                    sumVal +=self.alpha[t-1,j]*self.A[j,i]
                self.alpha[t,i] = sumVal*self.B[i,self.observeQueue[t]]

    def backWard(self): #后向处理
        self.beta = mat(zeros((self.tNum,self.N)))
        #公式10.19
        self.beta[self.tNum-1,:] = 1
        #公式10.20
        for t in range(self.tNum-2,-1,-1): #倒序遍历，从tNum-2开始
            for i in range(self.N):
                self.beta[t,i] = 0.0
                for j in range(self.N):
                    self.beta[t,i] += self.A[i,j]*self.B[j,self.observeQueue[t+1]]*self.beta[t+1,j]

    def calGamma(self,i, t):  #给定模型lamda和观测O，在时刻t处于状态i的概率
        #公式10.24
        num = self.alpha[t,i]* self.beta[t,i]
        den = self.alpha[t,:]*self.beta[t,:].T
        return num/den
    def calKsi(self,t,i,j):
        #公式10.26
        num = self.alpha[t,i]*self.A[i,j]*self.B[j,self.observeQueue[t+1]]*self.beta[t+1,j]
        den = 0.0
        for i in range(self.N):
            for j in range(self.N):
                den += self.alpha[t,i]*self.A[i,j]*self.B[j,self.observeQueue[t+1]]*self.beta[t+1,j]
        return num/den

    def randomInit(self):#随机生成A和B
        #A，转移概率矩阵，每一行的概率和为1
        for i in range(self.N):
            self.A[i,:] = [random.randint(0, 100) for t in range(self.N)]
            self.A[i,:] /= sum(self.A[i,:])
        #B，观测概率矩阵，每一行的概率和为1
        for i in range(self.N):
            self.B[i, :] = [random.randint(0, 100) for t in range(self.M)]
            self.B[i, :] /= sum(self.B[i, :])
        #pai,初始化相同，表明个状态出现的概率相同
        self.pai = mat(1.0 / self.N * ones((self.N, 1)))
    def train(self,observeQueue,maxSteps = 10):
        self.tNum = len(observeQueue)  #观测队列的元素数目
        self.observeQueue = observeQueue  #观测序列
        self.alpha = mat(zeros((self.tNum,self.N))) #alpha

        step = 0
        while step<maxSteps:
            step += 1
            temp_A = mat(zeros((self.N,self.N)))
            temp_B = mat(zeros((self.N,self.M)))
            temp_pai = mat(zeros((self.N,1)))

            self.forward()
            self.backWard()
            #计算A[i,j]转移概率矩阵
            for i in range(self.N):
                for j in range(self.N):
                    num = 0.0
                    den = 0.0
                    for k in range(self.tNum-1):
                        num += self.calKsi(k,i,j)
                        den += self.calGamma(i,k)
                    temp_A[i,j] = num/den
            #计算B[j,k]观测概率矩阵
            for j in range(self.N):
                for k in range(self.M):
                    num = 0.0
                    den = 0.0
                    for t in range(self.tNum):
                        if k == self.observeQueue[t]:  #因为状态全部用数字表示的，k为其中一个状态，若是其他表示，此处需要处理
                            num += self.calGamma(j, t)
                        den += self.calGamma(j, t)
                    temp_B[j,k] = num / den
            # pi_i
            for i in range(self.N):
                temp_pai[i,0] = self.calGamma(i, 0)

            self.A = temp_A
            self.B = temp_B
            self.pai = temp_pai
    def generateQueue(self,length):
        import random
        randNum = random.randint(0,1000)/1000.0
        i = 0
        I = []

        #初始状态
        while(self.pai[i]<randNum or self.pai[i]<1e-6):
            #当选择概率大于1e-6且随机的概率小于self.pai[i]的状态作为初始状态
            randNum -= self.pai[i]
            i += 1
        I.append(i)  #这里的状态用数字表示，也可以是其他形式

        #生成状态序列
        for i in range(1,length):
            lastI = I[-1]
            randNum = random.randint(0, 1000) / 1000.0
            j = 0
            while(self.alpha[lastI,j] < randNum or self.alpha[lastI,j]<1e-4):
                randNum -= self.alpha[lastI,j]
                j += 1
                if(j==self.N-1):
                    break
            I.append(j)

        #生成观测序列
        Y = []
        for i in range(length):
            randNum = random.randint(0,1000)/1000.0
            k = 0
            while(self.B[I[i],k]<randNum or self.B[I[i],k] < 1e-4):
                randNum -= self.B[I[i],k]
                k += 1
                if k == self.M-1:
                    break
            Y.append(k)
            return Y
        def predictViterbi():
            delta = mat(zeros((self.N,self.tNum)))
            delta[0,:] = multiply(self.pai.T,self.B[:,self.observeQueue[0]])
            psai = mat(zeros((self.N,self.tNum)))
            psai[0,:] = 0
            for t in range(1,self.tNum):
                for i in range(self.N):
                    delta[t,i] = multiply(max(multiply(delta[t-1,:].T,self.A[:,i].T)),self.B[i,self.observeQueue[t-1]])
                    psai[t,i] = argmax(multiply(delta[t-1,:].T,self.A[:,i].T))
            P = max(delta[self.tNum-1,:])
            it = argmax(delta[self.tNum-1,:])

            I = []
            I.append(it)
            for i in range(self.tNum-2,-1,-1):
                it = psai[i+1,I[0]]
                I.insert(0,it)
            return I

