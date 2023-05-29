import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.conv4 = GCNConv(num_features_xd*4, num_features_xd * 8)
        self.conv5 = GCNConv(num_features_xd*8, num_features_xd * 16)
        self.fc_g1 = torch.nn.Linear(num_features_xd*16, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Dropout：在不同的训练过程中随机扔掉一部分神经元，为了防止过拟合或减轻过拟合，一般用在全连接层
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)



    def forward(self, data):  # 向前传播
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        t = data.T
        t = t[:, None]  # 把tensor从（1，)变成（1，1）
        p = data.P
        p = p[:, None]

        x = self.conv1(x, edge_index)  # 第一层启动运算，输入为节点及特征和边的系数矩阵
        x = self.relu(x)  # 激活函数

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = self.conv4(x, edge_index)
        x = self.relu(x)   
        
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        
        x = gmp(x, batch)       # 池化降维，根据batch的值知道有多少张图片，再将每张图片的节点取一个全局最大的节点作为该张图片的一个输出

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # add some dense layers
        xc = self.fc1(x)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
