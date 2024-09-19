import torch
import math
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.conv import WLConvContinuous, ChebConv, SSGConv
import torch_geometric as pyg
import torch.nn.functional as F
import torch_scatter
from torch import nn

class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SchInteractionNetwork(pyg.nn.MessagePassing): # SchInteractionNetwork class
    """Interaction Network as proposed in this paper: 
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html
    SchNet as proposed in this paper:
    https://arxiv.org/abs/1706.08566"""
    
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)
        self.spectralConv = SSGConv(in_channels=hidden_size, out_channels=hidden_size, alpha=0.7)

    def forward(self, x, edge_index, edge_feature, node_dist):
        laplacian = pyg_utils.get_laplacian(edge_index)
        node = self.spectralConv(x=x, edge_index=laplacian[0], edge_weight=laplacian[1])
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature, node_dist=node_dist)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out + node
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature, node_dist):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, node_dist, dim_size=None):
        #out = torch_scatter.scatter(torch.div(inputs, 1+node_dist), index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
        out = torch_scatter.scatter(torch.mul(inputs,node_dist), index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
        return (inputs, out)

class GatNetwork(pyg.nn.MessagePassing): # GAT Class

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GatNetwork, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = torch.nn.Linear(self.in_channels, self.heads * self.out_channels)
        self.lin_r = self.lin_l
        self.att_l = torch.nn.Parameter(torch.randn(self.heads, self.out_channels))
        self.att_r = torch.nn.Parameter(torch.randn(self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        out = self.propagate(edge_index, x=(x_l,x_r), alpha=(alpha_l,alpha_r), size=size).reshape(-1, H*C)
        out = out.reshape(-1, H, C).mean(dim=1)
        return out
    
    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        if ptr is not None:
          alpha = pyg.utils.softmax(alpha, ptr)
        else:
          alpha = pyg.utils.softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout)
        out = x_j * alpha
        return out

    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return out

class EGatNetwork(pyg.nn.MessagePassing):  # EGAT class

    def __init__(self, in_node_channels, in_edge_channels, out_channels, heads = 3, layers = 3, bias = True, get_attn = False, use_F = True,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(EGatNetwork, self).__init__(node_dim=0, **kwargs)

        self.in_node_channels = in_node_channels
        self.in_edge_channels = in_edge_channels
        self.out_channels = out_channels
        self.heads = heads
        self.get_attn = get_attn
        self.use_F = use_F
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.node_out = None
        self.edge_out = None
        self.attn_weights = None
        
        # linear transformation layers for node and edge features
        # self.lin_node = torch.nn.Linear(self.in_node_channels, self.out_channels, bias=True)
        # self.lin_edge = torch.nn.Linear(self.in_edge_channels, self.out_channels, bias=True)
        self.lin_node_i = torch.nn.Linear(self.in_node_channels, self.heads * self.out_channels, bias=False)
        self.lin_node_j = torch.nn.Linear(self.in_node_channels, self.heads * self.out_channels, bias=False)
        self.lin_edge_ij = torch.nn.Linear(self.in_edge_channels, self.heads * self.out_channels, bias=False)

        # attention MLP to multiply with transformed node and edge features 
        self.attn_A = MLP(3*self.heads*self.out_channels, self.heads*self.out_channels, self.heads*self.out_channels, layers)
        #self.attn_A = torch.nn.Linear(3*self.heads*self.out_channels, self.heads*self.out_channels)

        # attention layer to multiply with new edge feature to get unnormalized attention weights
        self.attn_F =  torch.nn.Parameter(torch.FloatTensor(size=(1, self.heads, self.out_channels)))

        # MLPS for compressing multi-head node and edge features
        self.node_mlp = MLP(self.heads * self.out_channels, self.out_channels, self.out_channels, layers)
        self.edge_mlp = MLP(self.heads * self.out_channels, self.out_channels, self.out_channels, layers)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.lin_node.weight)
        # nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_node_i.weight)
        nn.init.xavier_uniform_(self.lin_node_j.weight)
        nn.init.xavier_uniform_(self.lin_edge_ij.weight)
        nn.init.xavier_uniform_(self.attn_F)

    def forward(self, h, edge_index, edge_feature, size=None):
        H, C = self.heads, self.out_channels
        h_prime_i = self.lin_node_i(h)                                           # shape [N,H*C]
        h_prime_j = self.lin_node_j(h)                                           # shape [N,H*C]
        f_ij = self.lin_edge_ij(edge_feature)                                    # shape [E,H*C]
        node_out = self.propagate(edge_index, x=(h_prime_i,h_prime_j), size=size, f_ij=f_ij)   # new multi-head node features
        self.node_out = self.node_mlp(node_out.reshape(-1,H*C))
        self.edge_out = self.edge_mlp(self.edge_out.reshape(-1,H*C)) #self.lin_edge(edge_feature) + self.edge_mlp(self.edge_out.reshape(-1,H*C))
        #self.node_out = self.lin_node(h) + self.node_out
        if self.get_attn:
          return self.node_out, self.edge_out, self.attn_weights
        else:
          return self.node_out, self.edge_out
    
    def message(self, x_i, x_j, index, ptr, size_i, f_ij):
        f_prime_ij = torch.cat([x_i, f_ij, x_j], dim=-1)                         # shape [E,H*C]
        f_prime_ij = self.attn_A(f_prime_ij)
        f_prime_ij = F.leaky_relu(f_prime_ij, negative_slope=self.negative_slope).reshape(-1, self.heads, self.out_channels) 
        self.edge_out = f_prime_ij                                               # new multi-head edge features
        eps = (f_prime_ij * self.attn_F) if self.use_F else f_prime_ij
        eps = eps.sum(dim=-1).unsqueeze(-1)                                      # unnormalized attention weights
        alpha = pyg_utils.softmax(eps, index, ptr, size_i)                       # normalized attention weights
        alpha = F.dropout(alpha, p=self.dropout) 
        self.attn_weights = alpha                                                # shape [E,H,1]
        out = x_j.reshape(-1,self.heads,self.out_channels) * alpha 
        return out

    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return out

