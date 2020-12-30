import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class Deep(nn.Module):
    def __init__(self, input_dim, deep_layers):
        super(Deep, self).__init__()

        deep_layers.insert(0, input_dim)
        deep_ayer_list = []
        for layer in list(zip(deep_layers[:-1], deep_layers[1:])):
            deep_ayer_list.append(nn.Linear(layer[0], layer[1]))
            deep_ayer_list.append(nn.BatchNorm1d(layer[1], affine=False))
            deep_ayer_list.append(nn.ReLU(inplace=True))
        self._deep = nn.Sequential(*deep_ayer_list)

    def forward(self, x):
        out = self._deep(x)
        return out

class Cross(nn.Module):
    """
    the operation is this module is x_0 * x_l^T * w_l + x_l + b_l for each layer, and x_0 is the init input
    """
    def __init__(self, input_dim, num_cross_layers):
        super(Cross, self).__init__()
        
        self.num_cross_layers = num_cross_layers
        weight_w = []
        weight_b = []
        batchnorm = []
        for i in range(num_cross_layers):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            batchnorm.append(nn.BatchNorm1d(input_dim, affine=False))

        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.bn = nn.ModuleList(batchnorm)

    def forward(self, x):
        out = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.num_cross_layers):
            # this compute mode is time-consuming.
            # out = torch.matmul(torch.bmm(x, torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2)), self.weight_w[i]) + self.weight_b[i] + out

            # use this compute mode to speed up calculation
            xxTw = torch.matmul(x, torch.matmul(torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2), self.weight_w[i].reshape(1, -1, 1)))
            xxTw = xxTw.reshape(xxTw.shape[0], -1)
            out = xxTw + self.weight_b[i] + out

            out = self.bn[i](out)
        return out

class DeepCross(BaseModel):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(DeepCross, self).__init__(config)
        self._config = config
        # 稠密和稀疏特征的数量
        self._num_of_dense_feature = dense_features_cols.__len__()
        self._num_of_sparse_feature = sparse_features_cols.__len__()

        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        # calculate all the embedding dimension of all the sparse features
        self.embedding_dims = list(map(lambda x: int(6 * pow(x, 0.25)), sparse_features_cols))
        # create embedding layers for all the sparse features
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=e[0], embedding_dim=e[1], scale_grad_by_freq=True) for e in list(zip(sparse_features_cols, self.embedding_dims))
        ])

        self._input_dim = self._num_of_dense_feature + sum(self.embedding_dims)

        self._deepNet = Deep(self._input_dim, self._config['deep_layers'])
        self._crossNet = Cross(self._input_dim, self._config['num_cross_layers'])

        self._final_dim = self._input_dim + self._config['deep_layers'][-1]
        self._final_linear = nn.Linear(self._final_dim, 1)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self._num_of_dense_feature], x[:, self._num_of_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        # 求出稀疏特征的隐向量
        sparse_embeds = [self.embedding_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        # 将dense特征和sparse特征聚合起来
        input = torch.cat([sparse_embeds, dense_input], axis=-1)

        deep_out = self._deepNet(input)
        cross_out = self._crossNet(input)

        final_input = torch.cat([deep_out, cross_out], dim=1)
        output = self._final_linear(final_input)
        output = torch.sigmoid(output)
        return output
