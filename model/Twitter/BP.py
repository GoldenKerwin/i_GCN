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

    def caculation(self, adjacency_matrix):
        num_nodes = adjacency_matrix.size(0)
        adjacency_no_self_loops = adjacency_matrix.clone()
        adjacency_no_self_loops[range(num_nodes), range(num_nodes)] = 0
        adjacency_no_self_loops = adjacency_no_self_loops.float()

        sibling_matrix = torch.matmul(adjacency_no_self_loops, adjacency_no_self_loops) + adjacency_no_self_loops

        sibling_matrix[range(num_nodes), range(num_nodes)] = 0

        # mask_hadamard = sibling_matrix.unsqueeze(2).expand(-1, num_nodes, -1)

        mask_hadamard = sibling_matrix.unsqueeze(1).expand(-1, 1, -1)
        mask_father = adjacency_matrix.unsqueeze(1).expand(-1, 1, -1)

        neighbor_count = adjacency_no_self_loops.sum(dim=1, keepdim=True)
        neighbor_count = torch.max(neighbor_count, torch.ones_like(neighbor_count))

        return mask_father, neighbor_count, mask_hadamard


    def forward(self, node_features, adjacency_matrix, mask_father, neighbor_count, mask_hadamard):
        node_features = node_features.float()  # 将node_features移动到指定设备
        input_features = node_features.shape[1]
        linear = torch.nn.Linear(input_features, self.in_features)  # 将linear移动到指定设备
        node_features = linear(node_features)

        weight_features = torch.mm(node_features, self.weight)  # 将self.weight移动到指定设备
        num_nodes = weight_features.size(0)
        features_expanded = weight_features.unsqueeze(2).expand(-1, -1, num_nodes)
        features_transpose_expanded = weight_features.unsqueeze(0).expand(1, -1, -1).transpose(1, 2)
        

        all_hadamard = torch.mul(features_expanded, features_transpose_expanded)

        masked_hadamard = all_hadamard * mask_hadamard  # 将mask_hadamard移动到指定设备

        adjacency_matrix = adjacency_matrix.unsqueeze(0)
        masked_hadamard = masked_hadamard.transpose(0, 1).transpose(0, 2)
        same_father_nodes = torch.matmul(adjacency_matrix, masked_hadamard)  

        same_father_nodes = same_father_nodes.transpose(0, 2).transpose(0, 1)
        same_father_nodes = same_father_nodes * mask_father  
        
        sum_hadamard = same_father_nodes.sum(dim=2)
        out_features = sum_hadamard / pow(neighbor_count.to(device), 2)  
        torch.cuda.empty_cache()
        return out_features


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
