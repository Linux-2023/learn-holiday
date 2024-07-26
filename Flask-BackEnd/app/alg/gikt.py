"""
模型
"""
import math
import numpy as np
import torch
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell,Parameter,Transformer
from params import DEVICE
import torch.nn as nn
from mlp import MLP
import torch_scatter
from  layers import CentralityEncoding

class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features ,dif_s, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0.5, normalization='None', input_norm=False,diffusion=True):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features + out_features, out_features, out_features, mlp2_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                         dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout
        self.diffusion = diffusion
        self.dif_s = dif_s
    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges):
        N = X.shape[-2]
        if self.diffusion :
            X = torch.spmm(self.dif_s, X)
        X0 = X
        Xve = self.W1(X)[..., vertex, :] # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr)  # [E, C], reduce is 'mean' here as default
        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]
        X = Xv
        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # register_buffer参数不参与梯度下降，但又希望保存model的时候将其保存下
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(32, 200, 128)，batch size为32,200个问题，问题维度为128
        """
        # 将x和positional encoding相加
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class GIKT(Module):

    def __init__(self, num_question, num_skill, q_neighbors, s_neighbors, qs_table,degree_index,dif_s, agg_hops=3, emb_dim=64,
                 dropout=(0.2, 0.4), hard_recap=False, rank_k=10, pre_train=False):
        super(GIKT, self).__init__()
        self.model_name = "gikt"
        self.num_question = num_question
        self.num_skill = num_skill
        self.q_neighbors = q_neighbors
        self.s_neighbors = s_neighbors
        self.agg_hops = agg_hops
        self.qs_table = qs_table
        self.emb_dim = emb_dim
        self.hard_recap = hard_recap
        self.rank_k = rank_k
        self.dif_s = dif_s

        self.positional_encoding = PositionalEncoding(64, dropout=0)
        self.transformer = Transformer(d_model=64, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)
        self.pred = Linear(64, 1)

        if pre_train:
            # 使用预训练之后的向量
            _weight_q = torch.load(f='data/q_embedding_128_150.pt', map_location=torch.device('cpu'))
            _weight_s = torch.load(f='data/s_embedding_128_150.pt', map_location=torch.device('cpu'))

            # num_question + num_skill：问题嵌入和技能嵌入
            # 后面num_question代表由超边抽象成的节点嵌入
            self.emb_table_node = nn.Embedding(num_question + num_skill + num_question, emb_dim)

            new_weight = self.emb_table_node.weight.data.clone()

            # 初始化权重（问题嵌入和技能嵌入部分）
            new_weight[:num_question] = _weight_q.detach().clone()
            new_weight[num_question:num_question + num_skill] = _weight_s.detach().clone()
            self.emb_table_node.weight.data = new_weight

        else:

            self.emb_table_node = Embedding(num_question+num_skill+num_question, emb_dim)
        self.emb_table_response = Embedding(2, emb_dim) # 回答结果嵌入表


        self.conv = EquivSetConv(emb_dim, emb_dim, dif_s, mlp1_layers=2,
                                 mlp2_layers=2,
                                 mlp3_layers=2, alpha=0.5, aggr='mean',
                                 dropout=0.5, normalization='ln',
                                 input_norm=True, diffusion=True).to(DEVICE)
        # self.degree = degree_index
        # # degree:每个节点的入度与出度
        # self.centralit = CentralityEncoding(num_in_degree = 5, num_out_degree = 5, node_dim = 128).to(DEVICE)
        # 该部分将入度与出度加入嵌入中（效果不好，没有使用）

    def forward(self, question, response, mask, edge_index):
        # question: [batch_size, seq_len]
        # response: [batch_size, 1]
        # mask: [batch_size, seq_len] 和question一样的形状, 表示在question中哪些索引是真正的数据(1), 哪些是补零的数据(0)
        # 每一个在forward中new出来的tensor都要.to(DEVICE)
        batch_size, seq_len = question.shape # batch_size表示多少个用户, seq_len表示每个用户最多回答了多少个问题
        V, E = edge_index[0].to(DEVICE), edge_index[1].to(DEVICE)
        embedding_weights = self.emb_table_node.weight
        feature_matrix = embedding_weights.cpu().detach().numpy()
        feature_tensor = torch.tensor(feature_matrix).to(DEVICE)
        emb_question = self.conv(feature_tensor, V, E)
        # 将嵌入矩阵权重进行卷积扩散
        emb_table = Embedding(self.num_question + self.num_skill + self.num_question, self.emb_dim,
                               _weight=emb_question)
        # 使用卷积扩散后的嵌入矩阵emb_table
        emb_question = emb_table(question) #问题向量 [32,200,128]
        zero_column = torch.zeros(batch_size, 1, dtype=torch.int64).to(DEVICE)
        # 加入起始令牌0
        response = torch.cat((zero_column, response), dim=1)[:, :-1]
        emb_response = self.emb_table_response(response) #回答向量 [32,200,128]
        emb_question = self.positional_encoding(emb_question)
        emb_response = self.positional_encoding(emb_response)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(response.size()[-1]).to(DEVICE)
        src_key_padding_mask = GIKT.get_key_padding_mask(question).to(DEVICE)
        tgt_key_padding_mask = GIKT.get_key_padding_mask(response).to(DEVICE)
        out = self.transformer(emb_question, emb_response,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.pred(out)
        out = torch.sigmoid(out.squeeze(-1))

        return out



    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


