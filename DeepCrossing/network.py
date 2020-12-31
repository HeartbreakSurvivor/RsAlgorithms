import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True)

    def forward(self, x):
        out = self.linear2(torch.relu(self.linear1(x)))
        out += x
        out = torch.relu(out)
        return out

class DeepCrossing(nn.Module):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(DeepCrossing, self).__init__()
        self._config = config
        # 稠密特征的数量
        self._num_of_dense_feature = dense_features_cols.__len__()
        # 稠密特征
        self.sparse_features_cols = sparse_features_cols
        self.sparse_indexes = [idx for idx, num_feat in enumerate(self.sparse_features_cols) if num_feat > config['min_dim']]
        self.dense_indexes = [idx for idx in range(len(self.sparse_features_cols)) if idx not in self.sparse_indexes]

        # 对特征类别大于config['min_dim']的创建Embedding层，其余的直接加入Stack层
        self.embedding_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            nn.Embedding(num_embeddings=self.sparse_features_cols[idx], embedding_dim=config['embed_dim'])
                for idx  in self.sparse_indexes
        ])

        self.dim_stack = len(self.sparse_indexes) * config['embed_dim'] + len(self.dense_indexes) + self._num_of_dense_feature
        self.residual_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            ResidualBlock(self.dim_stack, layer) for layer in config['hidden_layers']
        ])

        self._final_linear = nn.Linear(self.dim_stack, 1)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self._num_of_dense_feature], x[:, self._num_of_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        sparse_embeds = [self.embedding_layers[idx](sparse_inputs[:, i]) for idx, i in enumerate(self.sparse_indexes)]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        # 取出sparse中维度小于config['min_dim']的Tensor
        indices = torch.LongTensor(self.dense_indexes)
        sparse_dense = torch.index_select(sparse_inputs, 1, indices)

        output = torch.cat([sparse_embeds, dense_input, sparse_dense], axis=-1)

        for residual in self.residual_layers:
            output = residual(output)

        output = self._final_linear(output)
        output = torch.sigmoid(output)
        return output

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
