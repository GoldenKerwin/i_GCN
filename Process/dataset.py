import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
#Data类是Geometric库中的一个类，用于存储图数据，包括节点特征、边索引、标签、根节点索引等。

class GraphDataset(Dataset):        #继承自Dataset类
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
                #..表示上一级目录，../表示上两级目录，../../表示上三级目录
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        #这行代码过滤出treeDic中大小在lower和upper之间的树，然后将它们的id存储在self.fold_x中
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x) #返回数据集大小

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        #数据文件的路径是data_path和id的组合，文件格式为.npz
        #allow_pickle=True表示允许读取.npz文件中的Python对象
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
            #对边索引按照droprate概率删除边
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
        #返回一个Data对象，包含节点特征、边索引、标签、根节点索引等信息;
        #torch.tensor()函数用于将numpy数组转换为张量

def collate_fn(data):
    return data 

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        #fold_x：一个列表，包含了图的ID。只保留那些在treeDic中存在且大小在lower和upper之间的图的ID。
        self.treeDic = treeDic
        #treeDic：一个字典，键是图的ID，值是图的结构。
        self.data_path = data_path
        #data_path：一个字符串，定义了图数据的存储路径。默认路径是’…/…/data/Weibograph’。
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        #tddroprate和budroprate：两个浮点数，定义了在数据增强过程中丢弃节点和边的概率。

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        #这个方法的作用是获取指定索引的图的数据，并进行dropout

        id =self.fold_x[index]
        #从fold_x列表中获取指定索引的图的ID
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        #从硬盘中加载指定ID的图的数据。数据文件的路径是data_path和图的ID拼接而成的。
        edgeindex = data['edgeindex']
        #这行代码从加载的数据中获取边的索引。
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        #对边索引按照tddroprate概率删除边
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        #这两行代码获取边的索引的转置
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        #对边索引按照budroprate概率删除边
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
        #这行代码返回一个Data对象，包含了图的节点特征、边的索引、边的索引的转置、图的标签、图的根节点和根节点的索引。

class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        #这四行代码获取边的索引和边的索引的转置。
        row.extend(burow)
        col.extend(bucol)
        #这两行代码将边的索引和边的索引的转置合并在一起。
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
