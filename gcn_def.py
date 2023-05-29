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
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Dropout：在不同的训练过程中随机扔掉一部分神经元，为了防止过拟合或减轻过拟合，一般用在全连接层

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        #一维卷积，只对宽度卷，不管高度
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(130, 1024)
        self.fc2 = nn.Linear(1024, 512)  #线性连接层，输入通道数为1024，输出通道数为512
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
        x = gmp(x, batch)       # 池化降维，根据batch的值知道有多少张图片，再将每张图片的节点取一个全局最大的节点作为该张图片的一个输出

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # protein input feed-forward:
        aa = torch.cat((t, p), 1)

        # flatten
        xt = aa.view(-1, 2)

        # concat
        xc = torch.cat((x, xt), 1)  #为全连接数据输入调整tensor的形状，由三维变为二维，分别表示批次大小和每批次数据长度
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
