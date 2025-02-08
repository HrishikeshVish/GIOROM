#!/usr/bin/env python
#-*- coding:utf-8 _*-
import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid

from Baselines.ode_model import Decoder, Derivative
from Baselines.network import MLP, SetEncoder


'''
    Cross Attention GPT neural operator
    Trunck Net: geom
'''
class PhysicsEngine(nn.Module):
    def __init__(self,
                 device,
                 state_dim,
                 code_dim,
                 hidden_c_enc,
                 n_layers,
                 coord_dim,

                 ):
        super(PhysicsEngine, self).__init__()
        self.window_size = 5
        self.device = device
        self.state_dim = state_dim
        self.code_dim = code_dim
        self.hidden_c_enc = hidden_c_enc
        self.n_layers = n_layers
        self.coord_dim = coord_dim
        net_dec_params = {
            'state_c': self.state_dim, 
            'code_c': self.code_dim, 
            'hidden_c': self.hidden_c_enc, 
            'n_layers': self.n_layers, 
            'coord_dim': self.coord_dim
            }
        self.engine = Decoder(**net_dec_params)

    def forward(self, data):
        #gs = dgl.unbatch(g)
        #gs = g
        #x = pad_sequence([_g.ndata['x'] for _g in gs]).permute(1, 0, 2)  # B, T1, F

        #Data preprocessing modified for Lagrangian dynamics datasets
        

        x_out = self.engine(data.pos, codes=data.recent_pos)
        return x_out