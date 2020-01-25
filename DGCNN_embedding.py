from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from hgnn_lib import HGNNLIB
from pytorch_util import weights_init, gnn_spmm


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU', latent_edge_feat_dim = None):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim
        
        
        latent_edge_feat_dim = 0  ###############################################################################33
        
        
        if latent_edge_feat_dim is None:
            latent_edge_feat_dim = num_node_feats

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
#         self.conv_params.append(nn.Linear(35, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i-1]))
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))
        
        self.conv_params.append(nn.Linear(latent_dim[-1], latent_dim[-1]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_edge_feat_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def forward(self, bihypergraph_list, node_feat, node_feat_, edge_feat):
        hypergraph_sizes = [(bihypergraph_list[i].num_nodes, bihypergraph_list[i].num_nodes_) for i in range(len(bihypergraph_list))]
        node_hdegs = [torch.Tensor(bihypergraph_list[i].hdegs) + 1 for i in range(len(bihypergraph_list))]
        node_hdegs = torch.cat(node_hdegs).unsqueeze(1)
        node_hdegs_ = [torch.Tensor(bihypergraph_list[i].hdegs_) + 1 for i in range(len(bihypergraph_list))]
        node_hdegs_ = torch.cat(node_hdegs_).unsqueeze(1)
        
        hyperedge_sizes = [torch.Tensor(bihypergraph_list[i].hyperedge_sizes) + 1 for i in range(len(bihypergraph_list))]
        hyperedge_sizes = torch.cat(hyperedge_sizes).unsqueeze(1)
        hyperedge_sizes_ = [torch.Tensor(bihypergraph_list[i].hyperedge_sizes_) + 1 for i in range(len(bihypergraph_list))]
        hyperedge_sizes_ = torch.cat(hyperedge_sizes_).unsqueeze(1)

        sp_n2m, sp_m2n, sp_n_2m_, sp_m_2n_, sp_m2m_, sp_m_2m,\
           _, _, _, _ = HGNNLIB.PrepareSparseMatrices(bihypergraph_list)
        
        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            
            sp_n2m = sp_n2m.cuda()
            sp_n_2m_ = sp_n_2m_.cuda()
            sp_m2n = sp_m2n.cuda()
            sp_m_2n_ = sp_m_2n_.cuda()
            sp_m2m_ = sp_m2m_.cuda()
            sp_m_2m = sp_m_2m.cuda()
            
            node_hdegs = node_hdegs.cuda()
            hyperedge_sizes = hyperedge_sizes.cuda()
            node_hdegs_ = node_hdegs_.cuda()
            hyperedge_sizes_ = hyperedge_sizes_.cuda()
        node_feat = Variable(node_feat)
        node_feat_ = Variable(node_feat_)
        
        
        edge_feat=None                        #########################################################
        
        
        
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
                
        sp_n2m = Variable(sp_n2m)
        sp_n_2m_ = Variable(sp_n_2m_)
        sp_m2n = Variable(sp_m2n)
        sp_m_2n_ = Variable(sp_m_2n_)
        sp_m2m_ = Variable(sp_m2m_)
        sp_m_2m = Variable(sp_m_2m)
        
        node_hdegs = Variable(node_hdegs)
        hyperedge_sizes = Variable(hyperedge_sizes)
        
        node_hdegs_ = Variable(node_hdegs_)
        hyperedge_sizes_ = Variable(hyperedge_sizes_)

        h = self.sortpooling_embedding(node_feat, edge_feat,
                                       sp_n2m, sp_n_2m_, sp_m2n, sp_m_2n_, sp_m2m_, sp_m_2m,
                                       hypergraph_sizes, node_hdegs, node_hdegs_,
                                       hyperedge_sizes, hyperedge_sizes_)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat,
                              sp_n2m, sp_n_2m_, sp_m2n, sp_m_2n_, sp_m2m_, sp_m_2m,
                              hypergraph_sizes, node_hdegs, node_hdegs_,
                              hyperedge_sizes, hyperedge_sizes_):
        ''' if exists edge feature, concatenate to node feature vector 
        S: sp_n2m --> n x m
        S^T: sp_m2n --> m x n
        S_: sp_n_2m_ --> n_ x m_
        S_^T: sp_m_2n_ --> m_ x n_
        B: sp_m2m_ --> m x m_
        B^T: sp_m_2m --> m_ x m
        X: node_feat --> n x d
        X_: edge_feat --> n_ x d_
        node_hdegs --> n x 1
        node_hdegs_ --> n_ x 1
        hyperedge_sizes --> m x 1
        hyperedge_sizes_ --> m_ x 1
        '''
        
        edge_feat=None           ###################################
        
        
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
#             input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(n2f_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        
        def unit_transformation(cur_message_layer, pre_mul, normalizer,conv_param_lv=None, nonlinearity = None):
            pool = gnn_spmm(pre_mul, cur_message_layer)  # Z = pre_mul * cur_message_layer
            if conv_param_lv is not None:
                linear = self.conv_params[conv_param_lv](pool)  # Z = Z * W
            else:
                linear=pool
            normalized_linear = linear.div(normalizer)  # Z = normalizer^-1 * Z
            if nonlinearity is None:
                return normalized_linear
            return nonlinearity(normalized_linear) # Z = nonlinearity(Z)
        
        node_mode = 0
        cat_message_layers = []
        cat_message_layers_ = []
        if node_mode == 0:
            cur_message_layer = node_feat # X
            lv = 0
            while lv < 2*len(self.latent_dim):
                # First node set: U (Authors)
                cur_message_layer = unit_transformation(cur_message_layer, sp_m2n, hyperedge_sizes, None, None)#m*d
                cur_message_layer = unit_transformation(cur_message_layer, sp_m_2m, hyperedge_sizes_, None, None)#m_*d
                cur_message_layer_ = unit_transformation(cur_message_layer, sp_n_2m_, node_hdegs_, lv, torch.tanh)#n_*c
                lv += 1
                cur_message_layer = unit_transformation(cur_message_layer_, sp_m_2n_, hyperedge_sizes_, None, None)#m_*c
                cur_message_layer = unit_transformation(cur_message_layer, sp_m2m_, hyperedge_sizes, None, None)#m*c
                cur_message_layer = unit_transformation(cur_message_layer, sp_n2m, node_hdegs, lv, torch.tanh)#n*c1
                lv += 1
                cat_message_layers.append(cur_message_layer)
                cat_message_layers_.append(cur_message_layer_)
            cur_message_layer = torch.cat(cat_message_layers, 1)
        

        ''' sortpooling layer '''
        num_bihypergraphs = len(hypergraph_sizes)
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(num_bihypergraphs, self.k, self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(len(hypergraph_sizes)):
            to_sort = sort_channel[accum_count: accum_count + hypergraph_sizes[i][node_mode]]
            k = self.k if self.k <= hypergraph_sizes[i][node_mode] else hypergraph_sizes[i][node_mode]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += hypergraph_sizes[i][node_mode]

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)

        to_dense = conv1d_res.view(num_bihypergraphs, -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)
