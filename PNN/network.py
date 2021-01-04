import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class InnerProduct(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
        product or inner product between feature vectors.
          Input shape
            - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
          Output shape
            - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape:
            ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.
          Arguments
            - **reduce_sum**: bool. Whether return inner product or element-wise product
    """
    def __init__(self, reduce_sum=True):
        super(InnerProduct, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, inputs):
        embed_list = inputs
        row,col = [], []
        num_inputs = len(embed_list)

        # 这里为了形成n(n-1)/2个下标的组合
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)
        # inner_product 中包含了 n(n-1)/2 个 embedding size大小的向量，为了减少计算复杂度，将最后的维度求和，即将embedding size大小变为1
        inner_product = p * q
        if self.reduce_sum:
            # 默认打开，将最后一维的数据累加起来，降低计算复杂度
            inner_product = torch.sum(inner_product, dim=2, keepdim=True)
        return inner_product

class OutterProduct(nn.Module):
    """
      Input shape
            - A list of N 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
            - 2D tensor with shape:``(batch_size,N*(N-1)/2 )``.
      Arguments
            - **filed_size** : Positive integer, number of feature groups.
            - **kernel_type**: str. The kernel weight matrix type to use,can be mat,vec or num
    """
    def __init__(self, field_size, embedding_size, kernel_type='mat'):
        super(OutterProduct, self).__init__()
        self.kernel_type = kernel_type

        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size

        if self.kernel_type == 'mat':
            self.kernel = nn.Parameter(torch.Tensor(embed_size, num_pairs, embed_size))
        elif self.kernel_type == 'vec':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))
        elif self.kernel_type == 'num':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))

        nn.init.xavier_uniform_(self.kernel)

    def forward(self, inputs):
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        # -------------------------
        if self.kernel_type == 'mat':
            p.unsqueeze_(dim=1)
            # k     k* pair* k
            # batch * pair
            kp = torch.sum(torch.mul(torch.transpose(torch.sum(torch.mul(p, self.kernel), dim=-1), 2, 1), q), dim=-1)
        else:
            # 1 * pair * (k or 1)
            k = torch.unsqueeze(self.kernel, 0)
            # batch * pair
            kp = torch.sum(p * q * k, dim=-1)
            # p q # b * p * k
        return kp

class PNN(BaseModel):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(PNN, self).__init__(config)
        # 稠密和稀疏特征的数量
        self._num_of_dense_feature = dense_features_cols.__len__()
        self._num_of_sparse_feature = sparse_features_cols.__len__()

        # create embedding layers for all the sparse features
        self.embedding_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            nn.Embedding(num_embeddings=sparse_features_cols[idx], embedding_dim=config['embed_dim']) for idx in range(self._num_of_sparse_feature)
        ])

        self.use_inner = config['use_inner']
        self.use_outter = config['use_outter']
        self.kernel_type = config['kernel_type']

        if self.kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")

        num_inputs = self._num_of_sparse_feature
        # 计算两两特征交互的总数
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)

        if self.use_inner:
            self.innerproduct = InnerProduct()
        if self.use_outter:
            self.outterproduct = OutterProduct(num_inputs, config['embed_dim'], kernel_type=config['kernel_type'])

        # 计算L1全连接层的输入维度
        if self.use_outter and self.use_inner:
            product_out_dim = 2*num_pairs + self._num_of_dense_feature + config['embed_dim'] * self._num_of_sparse_feature
        elif self.use_inner or self.use_outter:
            product_out_dim = num_pairs + self._num_of_dense_feature + config['embed_dim'] * self._num_of_sparse_feature
        else:
            raise Exception("you must specify at least one product operation!")

        self.L1 = nn.Sequential(
            nn.Linear(in_features=product_out_dim, out_features=config['L2_dim']),
            nn.ReLU()
        )
        self.L2 = nn.Sequential(
            nn.Linear(in_features=config['L2_dim'], out_features=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self._num_of_dense_feature], x[:, self._num_of_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        # 求出稀疏特征的隐向量
        sparse_embeds = [self.embedding_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # 线性信号lz
        linear_signal = torch.cat(sparse_embeds, axis=-1)

        sparse_embeds = [e.reshape(e.shape[0], 1, -1) for e in sparse_embeds]
        if self.use_inner:
            inner_product = torch.flatten(self.innerproduct(sparse_embeds), start_dim=1)
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        if self.use_outter:
            outer_product = self.outterproduct(sparse_embeds)
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        if self.use_outter and self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product, outer_product], dim=1)

        # 将dense特征和sparse特征聚合起来
        dnn_input = torch.cat([product_layer, dense_input], axis=-1)
        output = self.L1(dnn_input)
        output = self.L2(output)
        return output

