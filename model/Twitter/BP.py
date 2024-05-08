import torch
from torch.nn import Parameter
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.decomposition import PCA

class Interaction_GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Interaction_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        # self.linear = torch.nn.Linear(, in_features)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def caculation(self,node_features, adjacency_matrix):
        node_features = node_features.float()
        num_nodes = adjacency_matrix.size(0)
        adjacency_no_self_loops = adjacency_matrix.clone()
        adjacency_no_self_loops[range(num_nodes), range(num_nodes)] = 0
        adjacency_no_self_loops = adjacency_no_self_loops.float()

        sibling_matrix = torch.matmul(adjacency_no_self_loops, adjacency_no_self_loops) + adjacency_no_self_loops

        sibling_matrix[range(num_nodes), range(num_nodes)] = 0

        # mask_hadamard = sibling_matrix.unsqueeze(2).expand(-1, num_nodes, -1)

        mask_hadamard = sibling_matrix.unsqueeze(1).expand(-1, 1, -1)
        print("mask_hadamard shape: ", mask_hadamard.shape)
        mask_father = adjacency_matrix.unsqueeze(1).expand(-1, 1, -1)
        print("mask_father shape: ", mask_father.shape)

        neighbor_count = adjacency_no_self_loops.sum(dim=1, keepdim=True)
        neighbor_count = torch.max(neighbor_count, torch.ones_like(neighbor_count))

        return mask_father, neighbor_count, mask_hadamard


    def forward(self, node_features, adjacency_matrix, mask_father, neighbor_count, mask_hadamard):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 指定设备

        node_features = node_features.float().to(device)  # 将node_features移动到指定设备
        input_features = node_features.shape[1]
        linear = torch.nn.Linear(input_features, self.in_features).to(device)  # 将linear移动到指定设备
        node_features = linear(node_features)

        weight_features = torch.mm(node_features, self.weight.to(device))  # 将self.weight移动到指定设备
        print("weight_features shape: ", weight_features.shape)
        num_nodes = weight_features.size(0)
        features_expanded = weight_features.unsqueeze(2).expand(-1, -1, num_nodes)
        # features_expanded = features_expanded.transpose(1,2)
        # features_transpose_expanded = weight_features.unsqueeze(0).expand(num_nodes,-1,-1)
        features_transpose_expanded = weight_features.unsqueeze(0).expand(1, -1, -1).transpose(1, 2)
        
        print("features_expanded shape: ", features_expanded.shape)
        print("features_transpose_expanded shape: ", features_transpose_expanded.shape)

        # all_hadamard = features_expanded * features_transpose_expanded
        all_hadamard = torch.mul(features_expanded, features_transpose_expanded)
        print("all_hadamard shape: ", all_hadamard.shape)

        masked_hadamard = all_hadamard * mask_hadamard.to(device)  # 将mask_hadamard移动到指定设备
        print("masked_hadamard shape: ", masked_hadamard.shape)

        # same_father_nodes = torch.matmul(adjacency_matrix.to(device), masked_hadamard) * mask_father.to(device)  # 将adjacency_matrix和mask_father移动到指定设备
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
        print("adjacency_matrix shape: ", adjacency_matrix.shape)
        masked_hadamard = masked_hadamard.transpose(0, 1).transpose(0, 2)
        print("masked_hadamard shape: ", masked_hadamard.shape)
        same_father_nodes = torch.matmul(adjacency_matrix.to(device), masked_hadamard)  # 将adjacency_matrix和mask_father移动到指定设备
        print("same_father_nodes shape: ", same_father_nodes.shape)

        same_father_nodes = same_father_nodes.transpose(0, 2).transpose(0, 1)
        same_father_nodes = same_father_nodes * mask_father.to(device)  # 将adjacency_matrix和mask_father移动到指定设备
        print("same_father_nodes shape: ", same_father_nodes.shape)
        
        sum_hadamard = same_father_nodes.sum(dim=0)
        print("sum_hadamard shape: ", sum_hadamard.shape)
        out_features = sum_hadamard / pow(neighbor_count.to(device), 2)  # 将neighbor_count移动到指定设备
        return out_features


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
