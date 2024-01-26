import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GAT
from torch_geometric.nn import TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=-1)


class GATNN(nn.Module):
    # Graph Attention Network
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels))
        self.Linear = Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.005, weight_decay=5e-4
        )

    def foward(self, x, edge_X):
        for conv in self.convs[:-1]:
            x = conv(x, edge_X)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.5, training=self.dropout)

        return self.convs[-1](x, edge_X)

    def train(self, data, epoch):
        total_loss = 0
        self.optimizer.zero_grad()
        for i in range(epoch):
            for batch in data:
                x, edge_X = batch.x, batch.edge_index
                out = self.foward(x, edge_X)
                loss = nn.CrossEntropyLoss(out, batch.y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss





class Net(torch.nn.Module):
    def __init__(self,embed_dim = 128, different_ids = 37353):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=different_ids, embedding_dim=embed_dim)
        #self.item_embedding = torch.nn.Embedding(num_embeddings=clicks_df.item_id.max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x