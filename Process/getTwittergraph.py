# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed  #简化并行计算
from tqdm import tqdm  #进度条
import sys  
cwd=os.getcwd()  #获取当前路径
class Node_tweet(object):
    #这个类的作用是表示一个节点，每个节点有子节点、索引、单词、索引和父节点等属性
    
    def __init__(self, idx=None):
        self.children = []
        #这个列表用于存储节点的子节点。
        self.idx = idx
        self.word = []
        #这个列表用于存储节点的单词。
        self.index = []
        self.parent = None
        #这个变量用于存储节点的父节点。

#生成词频和词索引的列表
def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

#生成树结构
def constructMat(tree):
    #tree是字典，键是节点索引，值是另一个字典，包括节点的属性，有父节点索引，最大度数，最大路径长度，词频和词索引
    index2node = {}
    #创建了一个新字典，键是节点索引，值是节点对象
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    #获取每个节点的父节点和词向量，然后将词向量的信息储存在节点对象中
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        #vec中存储了词索引和词频
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        #如果有父节点，将父节点的信息存储在节点对象中，并将当前节点添加到父节点的子节点列表中
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        #如果是根节点，将其信息存储在节点对象中
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])  #创建一个全0矩阵
    if len(root_index)>0:
        #如果根节点有词向量，将词向量信息存储在根节点的特征矩阵中
        rootfeat[0, np.array(root_index)] = np.array(root_word)

    #创建邻接矩阵和收集与节点相关的信息
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
                #这段代码遍历 index2node 中的每个节点。
                #如果 index2node[index_i+1] 有子节点，并且 index2node[index_j+1] 是 index2node[index_i+1] 的子节点
                #那么在邻接矩阵的相应位置设置为1，并将行列索引添加到 row 和 col 列表中。
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex

#根据输入的单词和索引列表来创建一个特征矩阵
def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
            #将 x_word[i] 的值赋给矩阵 x 的第 i 行和 x_index[i] 列。
    return x

def main(obj):
    #读取和处理一些数据，然后将这些数据保存为 .npz 文件
    treePath = os.path.join(cwd, 'data/' + obj + '/data.TD_RvNN.vol_5000.txt')
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))
    #这段代码打开 treePath 指向的文件，并逐行读取。对于每一行，它会解析出一些信息，并将这些信息存储在 treeDic 字典中。

    labelPath = os.path.join(cwd, "data/" + obj + "/" + obj + "_label_All.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    #主要作用是读取和处理标签信息
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    #这五行代码初始化了两个空列表（event 和 y）、四个计数器（l1、l2、l3 和 l4）和一个空字典（labelDic）
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
    #检查标签是否在四个预定义的标签集合中。如果是，那么它会在 labelDic 字典中为相应的事件 ID 设置一个值，并增加相应的计数器
    print(len(labelDic))
    print(l1, l2, l3, l4)

    def loadEid(event,id,y):
        #处理事件数据，并将处理后的数据保存为 .npz 文件
        #event 是事件字典，id 是事件 ID，y 是事件标签
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            #调用 constructMat 函数处理 event 数据
            x_x = getfeature(x_word, x_index)
            #调用 getfeature 函数处理 x_word 和 x_index，并返回特征矩阵 x_x
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            #将 rootfeat，tree，x_x，rootindex，和 y 转换为 NumPy 数组
            np.savez( os.path.join(cwd, 'data/'+obj+'/'+obj+'graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            return None
    print("loading dataset", )
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    #这段代码使用并行计算来处理事件数据。它会遍历 event 列表，并调用 loadEid 函数处理每个事件。loadEid 函数会调用 constructMat 函数来处理事件数据，并调用 getfeature 函数来生成特征矩阵。然后它会将处理后的数据保存为 .npz 文件。
    return

if __name__ == '__main__':
    obj= sys.argv[1]
    #索引为 1，因为索引 0 是程序自身的名称
    main(obj)