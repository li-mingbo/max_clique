# -*- coding: utf-8 -*-
'''
Created on Dec 15, 2016
li is short for machine study
@author: li mingbo
密集无向图的团数算法(算法1)

设无向图的顶点数是n。如果用快速近似算法获得团数 k >= int(n/2+0.5), 适用此算法。

1, 算法1逻辑：
    a) 快速近似算法获得团数
    b) 把这个近似最大团的顶点排在前面，
    c) 计算剩下的顶点（n-k）的近似最大团数（记为m），同时把这些点排在k个点后面，如果近似最大团数 m 等于 n-k，进入 d) 步。否则
    暂不考虑，转为其它算法。{ 也许可以，把(n-k-m)个剩下的点，与它相连的点构成的子图去计算团数（一般算法，近似染色算法等）， 对于k+m个点组成的子图，进入d) 步。}

    d) 无向图的关联矩阵的k点子图（关联矩阵左上角），m点子图（关联矩阵右下角），这两者的关联关系（关联矩阵右上角，或左下角）。
        如下图（n=6, k=3,m=3）：
      0  1  1  1  1  0
      1  0  1  1  1  0
      1  1  0  0  0  1
               0  1  1
               1  0  1
               1  1  0

        首先设前3行的后3列为3*3的块，看有没有一行中3列全是1，有，则团数由3变为4，若没有，看有没有一列中3行全是1，有，则团数由3变为4，若没有，则使用如下算法（算法2）：
        匈牙利算法，依据是此图的补图是二分图，这个二分图的独立数就是团数，二分图的独立数=二分图的顶点数（m+n）-最大匹配数；为求出最大匹配最正规的方法是使用匈牙利算法，但为了计算速度，我们尝试新的算法（算法3）。

2, 算法3逻辑：
    a) 随机找到原图的补图较大的匹配记为块A，把它置换排序到块的左上角
    b) 找到跟块A不相交的块B，随机找到块B对应的匹配记为块C，把它置换排序到块A的右下角，合并到块A中，同时块B缩小
    c) 设块B中的某列对应点为y，某行对应点为x，如果，x的交叉的列序号q等于y的交叉的行序号的p相等（p=q）那么这两个点可以匹配。可以合并到A块中。
    d) 如果y的首行（0行）是1，因为可以跟首列（0列）对换，兑换后，按步骤c)做。
    e) 设块B中的某列对应点为y的第2行跟首行交换，按d) 步骤计算，重复尝试第三行 ...
    f) 尝试检验，计算独立数。B块中的点必是独立的，再跟其它点合起来的最大独立数k，如果k=m+n-目前最大匹配数。则成功。否则，使用算法2。

其它备忘： 在步骤e) 后，还可以尝试。x某列为1，跟对角线相交的左侧行中有1，则x行可以跟这行互换，互换后又可以执行c) - e)
'''
from numpy import *
import datetime
import os 
import sys
import json
import io
import copy

import numpy as np
import networkx as nx
import operator
import time
import networkx.algorithms as alg
import networkx.algorithms.approximation as appr

# function: get directory of current script, if script is built
#   into an executable file, get directory of the excutable file
def current_file_directory():
            import inspect
            path = os.path.realpath(sys.path[0])        # interpreter starter's path
            if os.path.isfile(path):                    # starter is excutable file
                path = os.path.dirname(path)
                return os.path.abspath(path)            # return excutable file's directory
            else:                                       # starter is python script
                caller_file = inspect.stack()[1][1]     # function caller's filename
                return os.path.abspath(os.path.dirname(caller_file))# return function caller's file's directory


def readTxtToList(pth):
    fst=True
    lst=[]
    with open(pth, 'r') as f0:
        lns=f0.readlines()
        for line in lns:
                if fst:
                    fst=False
                    continue
                s1=line.replace(' \r\n','')
                s1=s1.replace(' \n','')
                ss=s1.split(' ')
                ss = [int(i) for i in ss]
                # if(len(ss)>0):
                #     ss.pop(len(ss)-1)
                lst.append(ss)
    return lst
def outLb(n):
    n2 = n * n
    dt = mat(zeros((n2 , 1)))
    return dt

def outA(leftLst):
    # n = len(leftLst[0])
    dt = mat(leftLst)
    return dt

def createMfile(fileName, countTarget):
    # gname = '48'
    lst2=readTxtToList(fileName)

    lst3=[]#readTxtToList('my'+gname+'_2.txt')
    outSameList = []

# createMfile('my96ts.txt',12)
class MaxClique:
    def __init__(self, dataGraph):
        self.G = nx.Graph()

        data = dataGraph.lstGraphData
        n = len(data)
        for i in range(n):
            for j in range(i+1,n):
                if data[i][j] == 1:
                    self.G.add_edge(i, j)
        pass

    def approximateMaxClque(self):
        return appr.max_clique(self.G)


    def computeMaxClique(self, maxTime):
        mc = 0
        tm = time.time()
        for i in alg.find_cliques(self.G):
            m = len(i)
            if m>mc:
                # print(str(i))
                mc = m
            if not (maxTime is None):
                if time.time()-tm>maxTime:
                        print('MaxCliqueList:%s' % str(i))
                        break
        print('MaxCliqueCount:%d'%mc)
        return mc


    pass
class GraphData:
    """GraphData"""

    def __init__(self, data):
        # 原始图的邻接矩阵数据（一般不可以修改）
        self.lstGraphData = data
        n = len(self.lstGraphData)
        # 图的点(index, 图的原始顶点的边数)
        self.lstVertex = [(i, sum(self.lstGraphData[i])) for i in range(n)]
        # 图的子完全图的列表
        self.lstSubBlock = []

    '''
    a) 快速近似算法获得团数
    '''
    def simpleComputeTuanshu(self):
        self.simpleComputeTuanshu5(None)
        #根据边数排序
        # graph = self
        # srtIndex = [index for (index, value) in sorted(copy.deepcopy(graph.lstVertex), key=lambda s: s[1], reverse=True)]
        #
        # srtIndex2 = copy.deepcopy(srtIndex)
        #
        # allRtn = []
        # while True:
        #     #计算第一个顶点的团数
        #     srtIndex3 = copy.deepcopy(srtIndex2)
        #     rtn = self.aVertexTuanshu(srtIndex2, 0)
        #     if len(rtn)>0:
        #         allRtn.append(rtn)
        #     else:
        #         break
        #     # srtIndex2 = srtIndex3
        #     srtIndex2 = [itm for itm in srtIndex3 if not itm in rtn]
        #     #根据边数排序
        #     lstVertex = [(srtIndex2[i], sum(self.lstGraphData[srtIndex2[i]])) for i in range(len(srtIndex2))]
        #     srtIndex2 = [value[0] for value in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
        #     # srtIndex2 = [value[0] for (index, value) in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
        #
        # # 排序
        # #b) 把这个近似最大团的顶点排在前面，
        # # tp = allRtn[0]
        # # allRtn[0] = allRtn[2]
        # # allRtn[2] = tp
        # self.lstSubBlock = [value for value in sorted(allRtn, key=lambda s: len(s), reverse=True)]
        #
        # # self.lstSubBlock[0] = [1,5]
        # # self.lstSubBlock[1] = [0,4]
        # # self.lstSubBlock[2] = [3]
        # # self.lstSubBlock.append([2])
        # print('lstSubBlock: %s' % str([len(i) for i in self.lstSubBlock]))
        # # print('lstSubBlock detail: %s' % str(self.lstSubBlock))
        # return self.lstSubBlock

    def dumpCurrentGraph(self):
        n = len(self.lstGraphData)
        dt = mat(zeros((n, n)))
        all = []
        for i in self.lstSubBlock:
            all.extend(i)
        for i in range(n):
            for j in range(i+1,n):
                if self.lstGraphData[all[i]][all[j]]==1:
                    dt[i,j] = 1
                    dt[j, i] = 1

        print(dt.getA())
        # print(dt)
    def revmoveAdjcent(self, lstSubBlockRowIndex, lstSubBlockVal):
        self.lstSubBlock[lstSubBlockRowIndex].remove(lstSubBlockVal)
        for i, v in enumerate(self.lstSubBlock):
            if i==lstSubBlockRowIndex:
                continue
            v2 = copy.deepcopy(v)
            for k in v2:
                if self.lstGraphData[lstSubBlockVal][k] == 0:
                    v.remove(k)

        pass

    def simpleComputeTuanshu3(self, firstBlock):
        n = len(self.lstGraphData)
        secondBlock = [i for i in range(n) if not (i in firstBlock)]
        self.lstSubBlock = []
        self.lstSubBlock.append(firstBlock)
        self.lstSubBlock.extend(self.simpleComputeTuanshu2(secondBlock))

    def simpleComputeTuanshu5(self, iFirst):
        #根据边数排序
        graph = self
        if iFirst is None:
            srtIndex = [index for (index, value) in sorted(copy.deepcopy(graph.lstVertex), key=lambda s: s[1], reverse=True)]
        else:
            srtIndex = [i for i in range(len(graph.lstVertex))]

        srtIndex2 = copy.deepcopy(srtIndex)

        allRtn = []
        while True:
            #计算第一个顶点的团数
            srtIndex3 = copy.deepcopy(srtIndex2)
            if iFirst is None:
                rtn = self.aVertexTuanshu(srtIndex2, 0)
            else:
                rtn = self.aVertexTuanshu(srtIndex2, iFirst if len(allRtn)==0 else 0)
            if len(rtn)>0:
                allRtn.append(rtn)
            else:
                break
            # srtIndex2 = srtIndex3
            srtIndex2 = [itm for itm in srtIndex3 if not itm in rtn]
            #根据边数排序
            lstVertex = [(srtIndex2[i], sum(self.lstGraphData[srtIndex2[i]])) for i in range(len(srtIndex2))]
            srtIndex2 = [value[0] for value in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
            # srtIndex2 = [value[0] for (index, value) in sorted(lstVertex, key=lambda s: s[1], reverse=True)]

        # 排序
        #b) 把这个近似最大团的顶点排在前面，
        # tp = allRtn[0]
        # allRtn[0] = allRtn[2]
        # allRtn[2] = tp
        self.lstSubBlock = [value for value in sorted(allRtn, key=lambda s: len(s), reverse=True)]

        # self.lstSubBlock[0] = [1,5]
        # self.lstSubBlock[1] = [0,4]
        # self.lstSubBlock[2] = [3]
        # self.lstSubBlock.append([2])
        # print('lstSubBlock: %s' % str([len(i) for i in self.lstSubBlock]))
        # print('lstSubBlock detail: %s' % str(self.lstSubBlock))
        return self.lstSubBlock

    def simpleComputeTuanshu2(self, srtIndex):
        allRtn = []
        srtIndex2 = copy.deepcopy(srtIndex)
        while True:
            #计算第一个顶点的团数
            srtIndex3 = copy.deepcopy(srtIndex2)
            rtn = self.aVertexTuanshu(srtIndex2, 0)
            if len(rtn)>0:
                allRtn.append(rtn)
            else:
                break
            # srtIndex2 = srtIndex3
            srtIndex2 = [itm for itm in srtIndex3 if not itm in rtn]
            #根据边数排序
            lstVertex = [(srtIndex2[i], sum(self.lstGraphData[srtIndex2[i]])) for i in range(len(srtIndex2))]
            srtIndex2 = [value[0] for value in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
            # srtIndex2 = [value[0] for (index, value) in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
        return allRtn


    '''
    计算一个近似最大团数，
    参数：vertexIndexArrar，顶点的索引列表；nFirst，哪个点
    返回值：这个最大团的顶点索引列表
    '''
    def aVertexTuanshu(self, vertexIndexArrar, nFirst):
        rtn = []
        if len(vertexIndexArrar) ==1:
            rtn = vertexIndexArrar
            return rtn
        if len(vertexIndexArrar) ==0:
            return rtn
        nFirstUsed = nFirst
        while len(vertexIndexArrar)>0:
            n = vertexIndexArrar[nFirstUsed]
            nFirstUsed=0
            rtn.append(n)
            vertexIndexArrar.remove(n)

            vertexIndexArrar2 = copy.copy(vertexIndexArrar)
            for itm in vertexIndexArrar:
                if self.lstGraphData[n][itm] == 0:
                    vertexIndexArrar2.remove(itm)
            vertexIndexArrar=vertexIndexArrar2
        # self.createLinkedSubGraph(vertexIndexArrar, nFirst)

        return rtn

    '''
    算法3逻辑：
    a) 随机找到原图的补图较大的匹配记为块A，把它置换排序到块的左上角
    b) 找到跟块A不相交的块B，随机找到块B对应的匹配记为块C，把它置换排序到块A的右下角，合并到块A中，同时块B缩小
    c) 设块B中的某列对应点为y，某行对应点为x，如果，x的交叉的列序号q等于y的交叉的行序号的p相等（p=q）那么这两个点可以匹配。可以合并到A块中。
    d) 如果y的首行（0行）是1，因为可以跟首列（0列）对换，兑换后，按步骤c)做。
    e) 设块B中的某列对应点为y的第2行跟首行交换，按d) 步骤计算，重复尝试第三行 ...
    f) 尝试检验，计算独立数。B块中的点必是独立的，再跟其它点合起来的最大独立数k，如果k=m+n-目前最大匹配数。则成功。否则，使用算法2。
    算法3 实现
    '''
    '''
    入参：graph，图的对象；leftTopVertexArr，graph的邻接矩阵的第一组点，其行会与第2组点rightBottomVertex的列相交成块
            leftTopVertexArr，arrVertexIndex，rightBottomVertex是顶点的索引，例如 [3,1,2,0]代表第4个顶点排在第1位，第2个顶点排在第2位，第3个顶点排在第3位，第1个顶点排在第4位.
    返回：[[返回的部分leftTopVertexArr][返回的部分rightBottomVertex]]，
            len([返回的部分leftTopVertexArr])+len([返回的部分rightBottomVertex]) 就是此块关联的最大团数
    '''
    @staticmethod
    def maxBlockByMaxMatch(graph, leftTopVertexArr, rightBottomVertex):
        leftTopVertexArr = copy.deepcopy(leftTopVertexArr)
        rightBottomVertex = copy.deepcopy(rightBottomVertex)
        x = len(leftTopVertexArr)
        y = len(rightBottomVertex)
        #过滤每列全是1的
        # columnDel = []
        # for ind in rightBottomVertex:
        #     if sum([graph.lstGraphData[leftTopVertexArr[i]][ind] for i in range(x)])==x:
        #         columnDel.append(ind)
        # rightBottomVertex = [itm for itm in rightBottomVertex if not (itm in columnDel)]

        '''a) 随机找到原图的补图较大的匹配记为块A，把它置换排序到块的左上角。len(leftTopVertexArr)必须>=len(rightBottomVertex)'''
        rowColIs0 = []
        rowSet = set()
        colSet = set()
        for ind in rightBottomVertex:
            if  ind in colSet:
                continue
            for row in leftTopVertexArr:
                if row in rowSet:
                    continue
                if graph.lstGraphData[row][ind]==0:

                        rowSet.add(row)
                        colSet.add(ind)
                        rowColIs0.append((row,ind))
                        break
                        # if len(rowColIs0) >= len(rightBottomVertex):
                        #     break
            if len(rowColIs0) >= len(rightBottomVertex):
                break

        if len(rowColIs0) == len(rightBottomVertex):
            return [leftTopVertexArr, rightBottomVertex]   #ok
        else:
            '''算法2'''
            return xylMatch(graph, leftTopVertexArr, rightBottomVertex)

    def swap(self, k1, k2):
        tmp = self.lstSubBlock[k1]
        self.lstSubBlock[k1] = self.lstSubBlock[k2]
        self.lstSubBlock[k2] = tmp

    def sortSubBlock(self, ind, direct):
        m = len(self.lstSubBlock[ind])
        if direct=='L':
            k=ind-1
            while k>=0:
                if len(self.lstSubBlock[k])<m :
                    self.swap(k,k+1)
                else:
                    return k
                k-=1
            return 0
        else:
            k=ind+1
            n = len(self.lstSubBlock)
            while k<n:
                if len(self.lstSubBlock[k])>m :
                    self.swap(k-1,k)
                else:
                    return k-1
                k+=1
            return n-1

    def sortSubGraph(self):
        self.sortSubGraph2(True)

    def sortSubGraph3(self):
        i=1
        while True:
            n = len(self.lstSubBlock)
            if i>=n:
                break
            rightBottom = self.lstSubBlock[i]
            for j in range(i):
                # if (not doSimpleComputeTuanshu ) and j>0:
                #     break
                leftTop = self.lstSubBlock[j]
                dt = GraphData.maxBlockByMaxMatch(self, leftTop, rightBottom)
                if len(dt[0])>len(leftTop):
                    #用新的大团数子图代替老的
                    #检测dt[0]是否是完全子图
                    dt2 = self.simpleComputeTuanshu2(dt[0])

                    if len(dt2)==1:
                        self.lstSubBlock[j] = dt[0]
                        dt2 = self.simpleComputeTuanshu2(dt[1])
                        isFirst = True
                        for k in dt2:
                            if isFirst:
                                self.lstSubBlock[i] = k
                            else:
                                self.lstSubBlock.insert(i,k)
                            self.sortSubBlock(i, '')
                            isFirst = False
                        # self.sortSubBlock(i, '')
                    elif len(dt2)>1:
                        dt2 = [value for value in sorted(dt2, key=lambda s: len(s), reverse=True)]
                        if len(dt2[0])<=len(leftTop):
                            break
                        self.lstSubBlock[j] = dt2[0]
                        dt3=dt2.remove(dt2[0])
                        dt2 = self.simpleComputeTuanshu2(dt[1])
                        if dt3:
                            dt2.extend(dt3)
                        isFirst = True
                        for k in dt2:
                            if isFirst:
                                self.lstSubBlock[i] = k
                            else:
                                self.lstSubBlock.insert(i,k)
                            self.sortSubBlock(i, '')
                            isFirst = False

                    # self.lstSubBlock[j] = dt[0]
                    # i = self.sortSubBlock(j, 'L') - 1

                    break
            break


    def sortSubGraph2(self, doSimpleComputeTuanshu):
        '''把最大团的子图按降序排列'''
        if doSimpleComputeTuanshu:
            self.simpleComputeTuanshu()
        else:
            self.lstSubBlock = [value for value in sorted(self.lstSubBlock, key=lambda s: len(s), reverse=True)]

        i=1
        while True:
            n = len(self.lstSubBlock)
            if i>=n:
                break
            rightBottom = self.lstSubBlock[i]
            for j in range(i):
                # if (not doSimpleComputeTuanshu ) and j>0:
                #     break
                leftTop = self.lstSubBlock[j]
                dt = GraphData.maxBlockByMaxMatch(self, leftTop, rightBottom)
                if len(dt[0])>len(leftTop):
                    #用新的大团数子图代替老的
                    #检测dt[1]是否是完全子图
                    dt2 = self.simpleComputeTuanshu2(dt[1])
                    if len(dt2) ==1 or len(dt2)==0:
                        self.lstSubBlock[i] = dt[1]
                        self.sortSubBlock(i, '')
                    else:
                        isFirst = True
                        for k in dt2:
                            if isFirst:
                                self.lstSubBlock[i] = k
                            else:
                                self.lstSubBlock.insert(i,k)
                            self.sortSubBlock(i, '')
                            isFirst = False

                    self.lstSubBlock[j] = dt[0]
                    i = self.sortSubBlock(j, 'L') - 1

                    break
            i+=1
            if i<=0:
                i=1

    '''
    计算
    '''
    def compute2PartTuanShu(self, selList):
        lstSubBlockBak = copy.deepcopy(self.lstSubBlock)
        block2 = []
        for i in range(1,len(self.lstSubBlock)):
            block2.extend(self.lstSubBlock[i])
        block3 = self.lstSubBlock[0]
        self.lstSubBlock = []
        self.lstSubBlock.append(block3)
        self.lstSubBlock.append(block2)

        nOld = len(block3)
        self.sortSubGraph3()
        if len(self.lstSubBlock[0])>nOld:
            return 'OK'

        selList.extend(self.lstSubBlock[0])
        self.lstSubBlock = lstSubBlockBak
        return 'Fail'
    # def createLinkedSubGraph(self, vertexIndexArrar, nFirst):
    #     n = vertexIndexArrar[nFirst][0]
    #     return [itm for itm in vertexIndexArrar if  itm[0]!=n and self.lstGraphData[n,itm[0]] == 1 ]

class GraphData2:
    """GraphData"""

    def __init__(self, data):
        # 原始图的邻接矩阵数据（一般不可以修改）
        self.lstGraphData = data
        n = len(self.lstGraphData)
        # 图的点(index, 图的原始顶点的边数)
        self.lstVertex = [(i, sum(self.lstGraphData[i])) for i in range(n)]
        # 图的子完全图的列表
        self.lstSubBlock = []

    '''
    a) 快速近似算法获得团数
    '''
    def simpleComputeTuanshu(self):
        self.simpleComputeTuanshu5(None)

    def revmoveAdjcent(self, lstSubBlockRowIndex, lstSubBlockVal):
        self.lstSubBlock[lstSubBlockRowIndex].remove(lstSubBlockVal)
        for i, v in enumerate(self.lstSubBlock):
            if i==lstSubBlockRowIndex:
                continue
            v2 = copy.deepcopy(v)
            for k in v2:
                if self.lstGraphData[lstSubBlockVal][k] == 0:
                    v.remove(k)


    def simpleComputeTuanshu3(self, firstBlock):
        n = len(self.lstGraphData)
        secondBlock = [i for i in range(n) if not (i in firstBlock)]
        self.lstSubBlock = []
        self.lstSubBlock.append(firstBlock)
        self.lstSubBlock.extend(self.simpleComputeTuanshu2(secondBlock))

    def simpleComputeTuanshu5(self, iFirst):
        #根据边数排序
        graph = self
        if iFirst is None:
            srtIndex = [index for (index, value) in sorted(copy.deepcopy(graph.lstVertex), key=lambda s: s[1], reverse=True)]
        else:
            srtIndex = [i for i in range(len(graph.lstVertex))]

        srtIndex2 = copy.deepcopy(srtIndex)

        allRtn = []
        while True:
            #计算第一个顶点的团数
            srtIndex3 = copy.deepcopy(srtIndex2)
            if iFirst is None:
                rtn = self.aVertexTuanshu(srtIndex2, 0)
            else:
                rtn = self.aVertexTuanshu(srtIndex2, iFirst if len(allRtn)==0 else 0)
            if len(rtn)>0:
                allRtn.append(rtn)
            else:
                break
            # srtIndex2 = srtIndex3
            srtIndex2 = [itm for itm in srtIndex3 if not itm in rtn]
            #根据边数排序
            lstVertex = [(srtIndex2[i], sum(self.lstGraphData[srtIndex2[i]])) for i in range(len(srtIndex2))]
            srtIndex2 = [value[0] for value in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
            # srtIndex2 = [value[0] for (index, value) in sorted(lstVertex, key=lambda s: s[1], reverse=True)]

        # 排序
        #b) 把这个近似最大团的顶点排在前面，
        # tp = allRtn[0]
        # allRtn[0] = allRtn[2]
        # allRtn[2] = tp
        self.lstSubBlock = [value for value in sorted(allRtn, key=lambda s: len(s), reverse=True)]

        # self.lstSubBlock[0] = [1,5]
        # self.lstSubBlock[1] = [0,4]
        # self.lstSubBlock[2] = [3]
        # self.lstSubBlock.append([2])
        # print('lstSubBlock: %s' % str([len(i) for i in self.lstSubBlock]))
        # print('lstSubBlock detail: %s' % str(self.lstSubBlock))
        return self.lstSubBlock

    def simpleComputeTuanshu2(self, srtIndex):
        allRtn = []
        srtIndex2 = copy.deepcopy(srtIndex)
        while True:
            #计算第一个顶点的团数
            srtIndex3 = copy.deepcopy(srtIndex2)
            rtn = self.aVertexTuanshu(srtIndex2, 0)
            if len(rtn)>0:
                allRtn.append(rtn)
            else:
                break
            # srtIndex2 = srtIndex3
            srtIndex2 = [itm for itm in srtIndex3 if not itm in rtn]
            #根据边数排序
            lstVertex = [(srtIndex2[i], sum(self.lstGraphData[srtIndex2[i]])) for i in range(len(srtIndex2))]
            srtIndex2 = [value[0] for value in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
            # srtIndex2 = [value[0] for (index, value) in sorted(lstVertex, key=lambda s: s[1], reverse=True)]
        return allRtn


    '''
    计算一个近似最大团数，
    参数：vertexIndexArrar，顶点的索引列表；nFirst，哪个点
    返回值：这个最大团的顶点索引列表
    '''
    def aVertexTuanshu(self, vertexIndexArrar, nFirst):
        rtn = []
        if len(vertexIndexArrar) ==1:
            rtn = vertexIndexArrar
            return rtn
        if len(vertexIndexArrar) ==0:
            return rtn
        nFirstUsed = nFirst
        while len(vertexIndexArrar)>0:
            n = vertexIndexArrar[nFirstUsed]
            nFirstUsed=0
            rtn.append(n)
            vertexIndexArrar.remove(n)

            vertexIndexArrar2 = copy.copy(vertexIndexArrar)
            for itm in vertexIndexArrar:
                if self.lstGraphData[n][itm] == 0:
                    vertexIndexArrar2.remove(itm)
            vertexIndexArrar=vertexIndexArrar2
        # self.createLinkedSubGraph(vertexIndexArrar, nFirst)

        return rtn

    '''
    算法x逻辑：
    a) 随机找到原图的最大的团记为块A，把它置换排序到左上角
    b) 其它的作为子图B，取子图B的第一个顶点v，查看v与A的连接关系
    c) 
    d) 
    e) 
    f) 
    算法x 实现
    '''
    '''
    入参：graph，图的对象；leftTopVertexArr，graph的邻接矩阵的第一组点，其行会与第2组点rightBottomVertex的列相交成块
            leftTopVertexArr，arrVertexIndex，rightBottomVertex是顶点的索引，例如 [3,1,2,0]代表第4个顶点排在第1位，第2个顶点排在第2位，第3个顶点排在第3位，第1个顶点排在第4位.
    返回：[[返回的部分leftTopVertexArr][返回的部分rightBottomVertex]]，
            len([返回的部分leftTopVertexArr])+len([返回的部分rightBottomVertex]) 就是此块关联的最大团数
    '''

    def maxBlockBySearch(self, leftTopVertexArr, rightBottomVertex):
        graph = self
        leftTopVertexArr = copy.deepcopy(leftTopVertexArr)
        rightBottomVertex = copy.deepcopy(rightBottomVertex)
        x = len(leftTopVertexArr)
        y = len(rightBottomVertex)
        #过滤每列全是1的
        # columnDel = []
        # for ind in rightBottomVertex:
        #     if sum([graph.lstGraphData[leftTopVertexArr[i]][ind] for i in range(x)])==x:
        #         columnDel.append(ind)
        # rightBottomVertex = [itm for itm in rightBottomVertex if not (itm in columnDel)]

        '''a) 随机找到原图的补图较大的匹配记为块A，把它置换排序到块的左上角。len(leftTopVertexArr)必须>=len(rightBottomVertex)'''
        rowColIs0 = []
        rowSet = set()
        colSet = set()
        for ind in rightBottomVertex:
            if  ind in colSet:
                continue
            for row in leftTopVertexArr:
                if row in rowSet:
                    continue
                if graph.lstGraphData[row][ind]==0:

                        rowSet.add(row)
                        colSet.add(ind)
                        rowColIs0.append((row,ind))
                        if len(rowColIs0) >= len(rightBottomVertex):
                            break
            if len(rowColIs0) >= len(rightBottomVertex):
                break

        if len(rowColIs0) == len(rightBottomVertex):
            return [leftTopVertexArr, rightBottomVertex]   #ok
        else:
            '''算法2'''
            return xylMatch(graph, leftTopVertexArr, rightBottomVertex)

    def swap(self, k1, k2):
        tmp = self.lstSubBlock[k1]
        self.lstSubBlock[k1] = self.lstSubBlock[k2]
        self.lstSubBlock[k2] = tmp

    def sortSubBlock(self, ind, direct):
        m = len(self.lstSubBlock[ind])
        if direct=='L':
            k=ind-1
            while k>=0:
                if len(self.lstSubBlock[k])<m :
                    self.swap(k,k+1)
                else:
                    return k
                k-=1
            return 0
        else:
            k=ind+1
            n = len(self.lstSubBlock)
            while k<n:
                if len(self.lstSubBlock[k])>m :
                    self.swap(k-1,k)
                else:
                    return k-1
                k+=1
            return n-1

    def sortSubGraph(self):
        self.sortSubGraph2(True)
    def sortSubGraph2(self, doSimpleComputeTuanshu):
        '''把最大团的子图按降序排列'''
        if doSimpleComputeTuanshu:
            self.simpleComputeTuanshu()
        else:
            self.lstSubBlock = [value for value in sorted(self.lstSubBlock, key=lambda s: len(s), reverse=True)]

        i=1
        while True:
            n = len(self.lstSubBlock)
            if i>=n:
                break
            rightBottom = self.lstSubBlock[i]
            for j in range(i):
                # if (not doSimpleComputeTuanshu ) and j>0:
                #     break
                leftTop = self.lstSubBlock[j]
                dt = GraphData.maxBlockByMaxMatch(self, leftTop, rightBottom)
                if len(dt[0])>len(leftTop):
                    #用新的大团数子图代替老的
                    #检测dt[1]是否是完全子图
                    dt2 = self.simpleComputeTuanshu2(dt[1])
                    if len(dt2) ==1 or len(dt2)==0:
                        self.lstSubBlock[i] = dt[1]
                        self.sortSubBlock(i, '')
                    else:
                        isFirst = True
                        for k in dt2:
                            if isFirst:
                                self.lstSubBlock[i] = k
                            else:
                                self.lstSubBlock.insert(i,k)
                            self.sortSubBlock(i, '')
                            isFirst = False

                    self.lstSubBlock[j] = dt[0]
                    i = self.sortSubBlock(j, 'L') - 1

                    break
            i+=1
            if i<=0:
                i=1
    '''
    根据一个顶点序列，以及一个顶点，求这个顶点与这个顶点连接的顶点组成的的子图
    '''
    # def createLinkedSubGraph(self, vertexIndexArrar, nFirst):
    #     n = vertexIndexArrar[nFirst][0]
    #     return [itm for itm in vertexIndexArrar if  itm[0]!=n and self.lstGraphData[n,itm[0]] == 1 ]

'''
主函数
'''
def computeTuanShu(fileName):
    #原始图的邻接矩阵数据
    lstGraphData = createRandGraphMetrix(41,1.2)#createRandGraphMetrixWithMaxClique(89,0.9, 23)#[array([0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0]), array([1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0]), array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]), array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]), array([0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]), array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]), array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1]), array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]), array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]), array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0])]
    #createRandGraphMetrix(11,0.7)#readTxtToList(fileName)
    # print('GraphData: %s' % str(lstGraphData))
    graph = GraphData(lstGraphData)

    mc = MaxClique(graph)

    # tm = time.time()
    # qu = mc.approximateMaxClque()
    # tm = time.time()-tm
    # print('time:%d, approximateMaxClque: len(%d), set:%s'%(tm, len(qu), str(qu)))
    qu=[]
    qmax=None
    tm = time.time()
    qu = mc.computeMaxClique(qmax)
    tm = time.time()-tm
    print('time:%d, realMaxClque: len(%d)'%(tm*1000,qu))

    tm = time.time()


    # graph.sortSubGraph()
    for iFirst in range(len(graph.lstGraphData)):
        graph.simpleComputeTuanshu5(iFirst)
        # print('lstSubBlock11 len: %s，data: %s' % (len(graph.lstSubBlock[0]),str(graph.lstSubBlock)))
        # graph.dumpCurrentGraph()
        # graph.sortSubGraph()
        graph.sortSubGraph2(False)
        # graph.dumpCurrentGraph()
        # print('lstSubBlock len: %s，data: %s' % (len(graph.lstSubBlock[0]),str(graph.lstSubBlock)))
        # checkValid(graph)
        maxBlock = len(graph.lstSubBlock[0])

        selList = []
        if len(graph.lstSubBlock[0])*2>=len(lstGraphData):
            '''特殊情况，可能可以直接确定团数'''
            result = graph.compute2PartTuanShu(selList)
            if result == 'OK':
                print('特殊情况 maxBlock111: %d,time:%d, v: %s' % (
                maxBlock, (time.time() - tm) * 1000, str(graph.lstSubBlock[0])))
                if qu == maxBlock:
                    return True
            # lstSubBlockBak = copy.deepcopy(graph.lstSubBlock)
            #
            # graph.lstSubBlock = lstSubBlockBak
            pass
        selList = []
        removeOne(graph, selList)
        # print('maxBlock2: %d, v: %s'%(len(selList),str(selList)))
        maxBlock = len(selList)
        # tm = time.time()-tm
        # maxBlock = len(graph.lstSubBlock[0])
        if not(qmax is None) and time.time()-tm > qmax:
            print(' maxBlock222: %d,time:%d, v: %s'%( maxBlock ,(time.time()-tm )*1000,str(graph.lstSubBlock[0])))
            return True


        if qu == maxBlock:
            print('ok, time:%d, turn:%d'%((time.time()-tm)*1000, iFirst))
            return True
        # else:
        #     return False
    # print('lstSubBlock: %s' % str([len(i) for i in graph.lstSubBlock]))

    # #再处理一次
    # selList.reverse()
    # graph.simpleComputeTuanshu3(selList)
    # # graph.sortSubGraph()
    # graph.sortSubGraph2(False)
    # checkValid(graph)
    # selList = []
    # removeOne(graph, selList)
    # print('maxBlock3: %d, v: %s'%(len(selList),str(selList)))
    # maxBlock222 = len(selList)
    # tm = time.time()-tm
    # # maxBlock = len(graph.lstSubBlock[0])
    # print('time3:%d, maxBlock3: %d, v: %s'%(tm, maxBlock222,str(graph.lstSubBlock[0])))

    # qu = maxBlock
    # if qu != maxBlock:
    #     if qu-maxBlock>2:
    #         raise Exception("Invalid maxBlock!", qu-maxBlock)
    #     return False
    print(' maxBlock333: %d,time:%d, v: %s'%( maxBlock ,(time.time()-tm)*1000 ,str(graph.lstSubBlock[0])))

    return False


def removeOne(graph, selList):
    lstSubBlockBak = copy.deepcopy(graph.lstSubBlock)
    n=0
    for i in graph.lstSubBlock:
        n += len(i)
        if n>0:
            break
    if n==0:
        return
    maxClq=len(graph.lstSubBlock[0]) if len(graph.lstSubBlock)>0 else 0
    sel = None
    for ind, i in enumerate(lstSubBlockBak):
        if ind>0:
            break
        for j in i:
            # graph.lstSubBlock[ind].remove(j)
            graph.revmoveAdjcent(ind, j)

            graph.sortSubGraph2(False)#
            k =len(graph.lstSubBlock[0]) if len(graph.lstSubBlock)>0 else 0
            if k>=maxClq:
                sel = j
                break
            graph.lstSubBlock = copy.deepcopy(lstSubBlockBak)
        if not (sel is None):
            break

    if sel is None:
        if len(graph.lstSubBlock)>0 and len(graph.lstSubBlock[0])>0:
            sel=graph.lstSubBlock[0][0]
            graph.revmoveAdjcent(0, sel)
            # del graph.lstSubBlock[0][0]
            selList.append(sel)
            # print('when none, sel:%d, lstSubBlock0:%d '%(sel,len(graph.lstSubBlock[0])))
    else:
        selList.append(sel)
        # print('when ok, sel:%d, lstSubBlock0:%d ' % (sel, len(graph.lstSubBlock[0])))
    removeOne(graph, selList)
    graph.lstSubBlock= copy.deepcopy(lstSubBlockBak)

def checkValid(graph):
    if len(graph.lstSubBlock) ==0:
        return
    for i in graph.lstSubBlock[0]:
        for j in graph.lstSubBlock[0]:
            if i==j:
                continue
            if graph.lstGraphData[i][j]!=1:
                print('lstSubBlock[0]不是完全图')
                return
    # print('lstSubBlock[0]是完全图')

def createRandGraphMetrix(nVertex,rateOf1):
    nOf1 = int(nVertex*nVertex*rateOf1)
    rtn = [np.zeros(nVertex, dtype=int) for i in range(nVertex)]
    import random

    for i in range(nOf1):
        n1 = random.randint(0,nVertex-1)
        n2 = random.randint(0,nVertex-1)
        if n1 == n2:
            continue
        rtn[n1][n2] = 1
        rtn[n2][n1] = 1

    return rtn

def createRandGraphMetrixWithMaxClique(nVertex,rateOf1, nVertexMaxClique):
    nOf1 = int(nVertex*nVertex*rateOf1)
    rtn = [np.zeros(nVertex, dtype=int) for i in range(nVertex)]
    import random

    for i in range(nOf1):
        n1 = random.randint(0,nVertex-1)
        n2 = random.randint(0,nVertex-1)
        if n1 == n2:
            continue
        rtn[n1][n2] = 1
        rtn[n2][n1] = 1

    m = int(nVertex/nVertexMaxClique)-1

    j=0
    arr=[]
    for i in range(nVertexMaxClique):
        j+=m
        arr.append(j)
    m = m/2
    if m==0:
        m=1
    for i in arr:
        for j in range(m,nVertex-m):
            rtn[i][j] = 0
            rtn[j][i] = 0
    for i in arr:
        for j in arr:
            rtn[i][j] = 1
            rtn[j][i] = 1

    return rtn

def xylMatch(graph, leftTopVertexArr, rightBottomVertex):
    '''

    :param graph: 图的邻接矩阵
    :param leftTopVertexArr: 左上完全图的点
    :param rightBottomVertex: 右下完全图的点
    :return: [new_leftTopVertexArr,new_rightBottomVertex], new_leftTopVertexArr是新的左上完全图的点（）
    '''
    maxVertex = len(leftTopVertexArr) #len(leftTopVertexArr)>=len(rightBottomVertex)
    # np.zeros(maxVertex, dtype=int)
    net = mat(zeros((maxVertex, maxVertex)))
    i=0
    for row in leftTopVertexArr:
        j = 0
        for col in rightBottomVertex:
            net[i, j] = graph.lstGraphData[row][col]
            j+=1
        i+=1

    ux, uy = np.zeros(maxVertex, dtype=int), np.zeros(maxVertex, dtype=int)
    result = np.zeros(maxVertex, dtype=int)
    for k in range(maxVertex):
        result[k] = -1
    sum = 0
    for i in range(maxVertex):
        RecUsed(ux, uy)
        if (match(i,ux,uy,result, maxVertex, net)):
            sum += 1
    if sum == maxVertex:
        '''团数不变'''
        return [leftTopVertexArr, rightBottomVertex]
    else:
        '''最大团数增大， 最大团数=所有定点数-最大匹配数'''
        n = maxVertex+maxVertex-sum
        '''寻找最大团，最大团必包含没有匹配的那几个点'''

        vertexRow1 = set()#[m for m in range(maxVertex)]
        vertexCol1 = set()
        vertexRow = set([m for m in range(maxVertex)])#
        vertexCol = set([m for m in range(maxVertex)])#
        for j in range(maxVertex):
            if result[j] != -1:
                vertexRow1.add(result[j])
                vertexCol1.add(j)

        vertexRow1 = vertexRow.difference(vertexRow1)#没有匹配的行，是最大团的元素
        vertexCol1 = vertexCol.difference(vertexCol1)#没有匹配的列，是最大团的元素
        for j in vertexRow1:
            for i in range(maxVertex):                #删除不连接的顶点
                if net[j,i]==0 and (i in vertexCol):
                    vertexCol.remove(i)
        for j in vertexCol1:
            for i in range(maxVertex):  # 删除不连接的顶点
                if net[i, j] == 0 and (i in vertexRow):
                    vertexRow.remove(i)

        leftTopVertexArrIn = vertexRow.difference(vertexRow1)#有匹配的行，此方法无效时使用
        rightBottomVertexIn = vertexCol.difference(vertexCol1)#有匹配的列，此方法无效时使用

        #再删除剩下的不相连的顶点，以左边为主
        vertexCol2 = copy.deepcopy(vertexCol)
        for i in vertexRow:
            for j in vertexCol:
                if net[i, j] == 0 and  (j in vertexCol2):
                    vertexCol2.remove(j)
        vertexCol = vertexCol2

        # if len(vertexRow1)+len(vertexCol1)+len(leftTopVertexArrIn) ==n or len(vertexRow1)+len(vertexCol1)+len(rightBottomVertexIn) ==n:
        #     left=[]
        #     right=[]
        #     for i in vertexRow1:
        #         left.append(leftTopVertexArr[i])
        #     for i in vertexCol1:
        #         left.append(rightBottomVertex[i])
        #     # left1=leftTopVertexArrIn
        #     # right1=rightBottomVertexIn
        #     if len(vertexRow1) + len(vertexCol1) + len(rightBottomVertexIn) == n:
        #         for i in rightBottomVertexIn:
        #             left.append(rightBottomVertex[i])
        #         for i in leftTopVertexArrIn:
        #             right.append(leftTopVertexArr[i])
        #     else:
        #         for i in rightBottomVertexIn:
        #             right.append(rightBottomVertex[i])
        #         for i in leftTopVertexArrIn:
        #             left.append(leftTopVertexArr[i])
        #
        #     print("method 1")
        #     return [left,right]
        # el
        if n==len(vertexCol)+len(vertexRow):#方法A有效
            left=[]
            right=[]
            for i in range(len(leftTopVertexArr)):
                left.append(leftTopVertexArr[i]) if i in vertexRow else right.append(leftTopVertexArr[i])
            for i in range(len(rightBottomVertex)):
                left.append(rightBottomVertex[i]) if i in vertexCol else right.append(rightBottomVertex[i])
            # print("method 2")
            return [left,right]
        else:#方法A无效，需要递归
            leftTopVertexArrIn = [leftTopVertexArr[itm] for itm in leftTopVertexArrIn]
            rightBottomVertexIn = [rightBottomVertex[itm] for itm in rightBottomVertexIn]
            if len(leftTopVertexArrIn)<len(rightBottomVertexIn):
                tmp=leftTopVertexArrIn
                leftTopVertexArrIn = rightBottomVertexIn
                rightBottomVertexIn = tmp
            r = xylMatch(graph, leftTopVertexArrIn, rightBottomVertexIn)
            if n == len(r[0])+len(vertexCol1)+len(vertexRow1):
                left = r[0]
                right = []
                for i in vertexRow1:
                    left.append(leftTopVertexArr[i])
                for i in vertexCol1:
                    left.append(rightBottomVertex[i])

                for i in rightBottomVertex:
                    if not( i in left):
                        right.append(i)
                for i in leftTopVertexArr:
                    if not( i in left):
                        right.append(i)
                # print("method 3")
                return [left,right] #right需要重新解析，可能需要拆成几个
            else:
                print('此算法存在错误，请检查算法逻辑')
            return None


def match(u, ux, uy, result, number, net):
    u = int(u)
    ux[u] = 1  # record node that was explored
    for v in range(number):
        if (uy[v] == 0 and net[u,v] == 0):
            uy[v] = 1

            if (result[v] == -1 or match(result[v], ux, uy, result, number, net)):
                result[v] = u
                return 1
    return 0
def RecUsed(ux, uy):  # Recover visited flag
    ux[:]=0
    uy[:]=0

# Number = 4
# Net = [[0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
# ux, uy = np.zeros(Number, dtype=int), np.zeros(Number, dtype=int)
# result = np.zeros(Number, dtype=int)
#

# def match(u):
#     #  global inc
#     u = int(u)
#     ux[u] = 1  # record node that was explored
#     for v in range(Number):
#         if (uy[v] == 0 and Net[u][v] == 1):
#             uy[v] = 1
#
#             if (result[v] == -1 or match(result[v])):
#                 result[v] = u
#                 return 1
#     return 0
#
#
# def RecUsed():  # Recover visited flag
#     global ux, uy
#     ux, uy = np.zeros(Number, dtype=int), np.zeros(Number, dtype=int)
#

if __name__ == '__main__':
    n = 0
    for i in range(1000):
        print("item i : %d" % i)
        print("fail n : %d"%n)
        if not computeTuanShu('myts6.txt'):
            n+=1
    # return
    # for k in range(Number):
    #     result[k] = -1
    # sum = 0
    # for i in range(Number):
    #     RecUsed()
    #     if (match(i)):
    #         sum += 1
    # print("Here are " + str(sum) + " pairs\n")
    # for j in range(Number):
    #     if (result[j] == -1):
    #         continue
    #     else:
    #         print(result[j], j)

'''
[array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]), array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]), array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]), array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]), array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]), array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]), array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])]
       
       
       
<type 'list'>: [array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
       1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]), array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]), array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]), array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]), array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]), array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]), array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]), array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
       0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]), array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]), array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]), array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1]), array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]), array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]), array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])]       
'''