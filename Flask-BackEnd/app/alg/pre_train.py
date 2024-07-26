"""
预训练问题和技能向量
"""
import math
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
from scipy import sparse
from params import DEVICE
from pebg import PEBG
from icecream import ic
from torch.utils.tensorboard import SummaryWriter

# 批量训练, 每次训练batch_size个问题,并值分析这些问题相关的技能
# 只取数据的批量的问题, 每次批量都将所有的技能考虑进计算
# model内的变量都是全部变量, 不带的是批量变量

qs_table = torch.tensor(sparse.load_npz('data/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
qq_table = torch.tensor(sparse.load_npz('data/qq_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
ss_table = torch.tensor(sparse.load_npz('data/ss_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
ss_table = (ss_table > 0).int()
qq_table = (qq_table > 0).int()
num_q = qs_table.shape[0]
num_s = qs_table.shape[1]
batch_size = 256
num_batch = math.ceil(num_q / batch_size)
hidden_dim =64    # hidden dim in PNN
keep_prob = 0.5

model = PEBG(qs_table, qq_table, ss_table, embed_dim=64,hidden_dim = 64,keep_prob = 0.5)
ic('开始训练模型')
# optimizer = torch.optim.Adam(params=list(model.parameters()) + list(), lr=0.001) # 优化器
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005)
loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
writer = SummaryWriter(log_dir='logs')


for epoch in range(150):
    model.train()
    train_loss = 0  # 总损失
    for idx_batch in range(num_batch):
        optimizer.zero_grad()  # 梯度清零
        # 计算索引
        idx_start = idx_batch * batch_size
        if idx_start == 0:
            idx_start = 1
        # 第一行是空白问题
        idx_end = min((idx_batch + 1) * batch_size, num_q)  # 结束索引,注意是否超过了问题数


        qs_target = model.qs_target[idx_start: idx_end]  # [size_batch, num_s]
        qq_target = model.qq_target[idx_start: idx_end]  # [size_batch, size_batch]
        ss_target = model.ss_target  # 与批量无关 [num_s, num_s]

        # 计算logit
        qs_logit, ss_logit, qq_logit ,p, pro_final_embed ,q_feature= model.forward(idx_start, idx_end,qs_target)

        # 计算损失

        loss_qs = loss(qs_logit, qs_target)  # L1
        loss_ss = loss(ss_logit, ss_target)
        loss_qq = loss(qq_logit, qq_target)  # L3

        pnn_mse = mse_loss(p, q_feature[idx_start:idx_end, -1]) #L4

        loss_sum = loss_qs + loss_qq + loss_ss + pnn_mse # 总损失

        train_loss += loss_sum.item()
        loss_sum.backward()  # 反向传播
        optimizer.step()  # 参数优化
    print(f'----------epoch: {epoch + 1}, train_loss: {train_loss}')
    writer.add_scalar(tag='pebg_loss_dim_128', scalar_value=train_loss, global_step=epoch)
    if epoch in [70,100,150]:
        model.eval()
        with torch.no_grad():
            qs_target = model.qs_target[0 : num_q]
            qs_logit, ss_logit, qq_logit, p, pro_final_embed, q_feature = model(0, num_q, qs_target)
            q_repre = model.q_embedding.weight.detach().cpu()
            s_repre = model.s_embedding.weight.detach().cpu()
            q_final_repre = pro_final_embed.detach().cpu()
        torch.save(q_final_repre, f'data/q_embedding_{hidden_dim}_{epoch}.pt')
        torch.save(s_repre, f'data/s_embedding_{hidden_dim}_{epoch}.pt')

writer.close()

