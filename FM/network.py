import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class FM(BaseModel):
    def __init__(self, config, p):
        super(FM, self).__init__(config)
        # 特征的个数
        self.p = p
        # 隐向量的维度
        self.k = config['latent_dim']
        # FM的线性部分，即∑WiXi
        self.linear = nn.Linear(self.p, 1, bias=True)
        # 隐向量的大小为nxk,即为每个特征学习一个维度为k的隐向量
        self.v = nn.Parameter(torch.randn(self.k, self.p), requires_grad=True)

    def forward(self, x):
        # 线性部分
        linear_part = self.linear(x)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v.t())
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return output

