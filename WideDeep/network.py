import torch
import torch.nn as nn
import torch.nn.functional as F

class Wide(nn.Module):
    def __init__(self, input_dim):
        super(Wide, self).__init__()
        # hand-crafted cross-product features
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)

class Deep(nn.Module):
    def __init__(self, config, hidden_layers):
        super(Deep, self).__init__()
        self.dnn = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_layers[:-1], hidden_layers[1:]))])
        self.dropout = nn.Dropout(p=config['deep_dropout'])

    def forward(self, x):
        for layer in self.dnn:
            x = layer(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x

class WideDeep(nn.Module):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(WideDeep, self).__init__()
        self._config = config
        # 稠密特征的数量
        self._num_of_dense_feature = dense_features_cols.__len__()
        # 稠密特征
        self.sparse_features_cols = sparse_features_cols

        self.embedding_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            nn.Embedding(num_embeddings = num_feat, embedding_dim=config['embed_dim'])
                for num_feat in self.sparse_features_cols
        ])

        # Deep hidden layers
        self._deep_hidden_layers = config['hidden_layers']
        self._deep_hidden_layers.insert(0, self._num_of_dense_feature + config['embed_dim'] * len(self.sparse_features_cols))

        self._wide = Wide(self._num_of_dense_feature)
        self._deep = Deep(config, self._deep_hidden_layers)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self._num_of_dense_feature], x[:, self._num_of_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        sparse_embeds = [self.embedding_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # sparse_embeds = []
        # for i in range(sparse_inputs.shape[1]):
        #     m = self.embedding_layers[i]
        #     y = m(sparse_inputs[:, i])
        #     sparse_embeds.append(y)

        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        # Deep模块的输入是稠密特征和稀疏特征经过Embedding产生的稠密特征的
        deep_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        wide_out = self._wide(dense_input)
        deep_out = self._deep(deep_input)

        assert (wide_out.shape == deep_out.shape)

        outputs = F.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
