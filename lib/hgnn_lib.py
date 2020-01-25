import ctypes
import numpy as np
import os
import sys
import torch
import pdb
import scipy

def combine_S(S_list):
    full_I = []
    full_J = []
    full_V = []
    I_prefix = 0
    J_prefix = 0
    list_I = []
    list_J = []
    for i, S in enumerate(S_list):
        n_rows, n_cols = S.shape
        I, J, V = scipy.sparse.find(S)
        full_I += list(I+I_prefix)
        full_J += list(J+J_prefix)
        full_V += list(V)
        list_I += [i]*n_rows
        list_J += list(range(I_prefix, I_prefix + n_rows))
        I_prefix += n_rows
        J_prefix += n_cols
    n2f_sp = torch.sparse.FloatTensor(torch.LongTensor([full_I, full_J]), torch.FloatTensor(full_V), torch.Size([I_prefix, J_prefix]))
    f2n_sp = torch.sparse.FloatTensor(torch.LongTensor([full_J, full_I]), torch.FloatTensor(full_V), torch.Size([J_prefix, I_prefix]))
    subhg_sp = torch.sparse.FloatTensor(torch.LongTensor([list_I, list_J]), torch.LongTensor([1]*len(list_I)), torch.Size([len(S_list), I_prefix]))
    return n2f_sp, f2n_sp, subhg_sp

class _hgnn_lib(object):

    def __init__(self):
        pass

    def PrepareSparseMatrices(self, bihypergraph_list):
        list_S = [x.S for x in bihypergraph_list]
        list_S_ = [x.S_ for x in bihypergraph_list]
        list_B = [x.B for x in bihypergraph_list]
        sp_n2m, sp_m2n, subhg_sp_n = combine_S(list_S)
        sp_n_2m_, sp_m_2n_, subhg_sp_n_ = combine_S(list_S_)
        sp_m2m_, sp_m_2m, subhg_sp_m = combine_S(list_B)
        _, _, subhg_sp_m_ = combine_S([x.T for x in list_B])
        return sp_n2m, sp_m2n, sp_n_2m_, sp_m_2n_, sp_m2m_, sp_m_2m,\
               subhg_sp_n, subhg_sp_n_, subhg_sp_m, subhg_sp_m_

HGNNLIB = _hgnn_lib()

