import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
import utils
import math
from .models import register
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
seed = 66
torch.manual_seed(seed)
np.random.seed(seed)


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=8, score_function='bi_linear', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True, beta=0.5):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.CrossAttention = Attention(embed_dim=512, hidden_dim=None, out_dim=None, n_head=8,
                                        score_function='bi_linear', dropout=0)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.1))

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

        self.beta = beta

    '''
    x_shot:[4,5,1,3,80,80]
    x_query:[4,75,3,80,80]

    '''

    def forward(self, clip_feat, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]  # [4,5,1]
        query_shape = x_query.shape[:-3]  # [4,75]
        img_shape = x_shot.shape[-3:]  # [3,80,80]

        x_shot = x_shot.view(-1, *img_shape)  
        x_query = x_query.view(-1, *img_shape)  
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot_fea, x_query_fea = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_query_fea = x_query_fea.view(*query_shape, -1) 

        fusion_fea, _ = self.CrossAttention(clip_feat.to(torch.float32), x_shot_fea.to(torch.float32))
        fusion_fea = fusion_fea.squeeze(dim=1)
        x_shot_refuse = self.fc(torch.cat([x_shot_fea, fusion_fea], dim=1))
        x_shot_refuse = x_shot_refuse.view(*shot_shape, -1)
        x_shot_fea = x_shot_fea.view(*shot_shape, -1)
        x_shot_refuse = x_shot_refuse * self.beta + x_shot_fea * (1 - self.beta)

        if self.method == 'cos':
            x_shot_refuse = x_shot_refuse.mean(dim=-2)
            x_shot_refuse = F.normalize(x_shot_refuse, dim=-1)
            x_query_fea = F.normalize(x_query_fea, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot_refuse = x_shot_refuse.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
            x_query_fea, x_shot_refuse, metric=metric, temp=self.temp)
        return logits
