"""
PEBG问题向量预训练模型
仅仅用于提取问题和技能之间的关系,不涉及具体用户
"""
import numpy as np
import torch.nn as nn
from icecream import ic
from params import *
from pnn import PNN

ic(torch.cuda.is_available())
ic(DEVICE)

q_feature = torch.tensor(np.load('data/q_feature.npy')).float().to(DEVICE)  # [num_q, size_q_feature] 问题属性特征,已生成



class PEBG(nn.Module):
    def __init__(self, qs_table, qq_table, ss_table, embed_dim, hidden_dim, keep_prob):
        super().__init__()
        num_q, num_s = qs_table.shape[0], qs_table.shape[1]

        self.qs_target = torch.as_tensor(qs_table, dtype=torch.float, device=DEVICE)  # [num_q, num_s] 问题-技能表
        self.qq_target = torch.as_tensor(qq_table, dtype=torch.float, device=DEVICE)  # [num_q, num_s] 问题-问题表
        self.ss_target = torch.as_tensor(ss_table, dtype=torch.float, device=DEVICE)  # [num_s, num_s] 技能-技能表
        self.q_diff = q_feature[:, :-1]  # [num_q, size_q_feature-1] 属性特征除了最后一位都可以作为难度特征 []

        diff_feat_num = q_feature.shape[1] - 1
        # self.q_embedding = nn.Parameter(torch.concat([self.qs_target, self.q_diff], dim=1)) # [num_q, num_s + 7]
        # self.s_embedding = nn.Parameter(self.ss_target) # [num_s, num_s]
        self.q_embedding = nn.Embedding(num_q, embed_dim).to(DEVICE)
        self.s_embedding = nn.Embedding(num_s, embed_dim).to(DEVICE)
        self.diff_feat_embedding = nn.Embedding(diff_feat_num, embed_dim).to(DEVICE)

        self.pnn = PNN(embed_dim, hidden_dim, keep_prob)
        self.apply(self.init_params)

    def init_params(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight.data, mean=0, std=0.1)

    def forward(self, b, e , qs_target):
        # 接收已经确定好批次的输入向量

        #qs_logit = torch.mm(q_embedding_fc, s_embedding_fc.T)  # [batch_size_q, num_s] 公式1
        #qq_logit = self.sigmoid[1](torch.matmul(q_embedding_fc, q_embedding_fc.T))  # [batch_size_q, size_batch] 公式5
        #ss_logit = self.sigmoid[2](torch.matmul(s_embedding_fc, s_embedding_fc.T))  # [num_s, num_s] 公式6

        qs_logit = torch.mm(self.q_embedding.weight[b:e], self.s_embedding.weight.T)  # (bs, skill_num)
        ss_logit = torch.mm(self.s_embedding.weight, self.s_embedding.weight.T)  # (skill_num, skill_num)
        qq_logit = torch.mm(self.q_embedding.weight[b:e], self.q_embedding.weight.T)  # (bs, pro_num)



        s_embedding = torch.mm(qs_target, self.s_embedding.weight) / torch.mean(qs_target, axis=1,keepdims=True)
        diff_feat_embedding = torch.mm(q_feature[b:e, :-1], self.diff_feat_embedding.weight)

        pro_final_embed, p = self.pnn([self.q_embedding.weight[b:e], s_embedding, diff_feat_embedding])

        return qs_logit, ss_logit, qq_logit, p, pro_final_embed,q_feature

