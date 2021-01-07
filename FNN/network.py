import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class FNN(BaseModel):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(FNN, self).__init__(config)
        # 稠密和稀疏特征的数量
        self.num_dense_feature = dense_features_cols.__len__()
        self.num_sparse_feature = sparse_features_cols.__len__()

        # FNN的线性部分，对应 ∑WiXi
        self.embedding_layers_1 = nn.ModuleList([
            nn.Embedding(num_embeddings=feat_dim, embedding_dim=1)
                for feat_dim in sparse_features_cols
        ])

        # FNN的Interaction部分，对应∑∑<Vi,Vj>XiXj
        self.embedding_layers_2 = nn.ModuleList([
            nn.Embedding(num_embeddings=feat_dim, embedding_dim=config['embed_dim'])
                for feat_dim in sparse_features_cols
        ])

        # FNN的DNN部分
        self.hidden_layers = [self.num_dense_feature + self.num_sparse_feature*(config['embed_dim']+1)] + config['dnn_hidden_units']
        self.dnn_layers = nn.ModuleList([
            nn.Linear(in_features=layer[0], out_features=layer[1])\
                for layer in list(zip(self.hidden_layers[:-1], self.hidden_layers[1:]))
        ])
        self.dnn_linear = nn.Linear(self.hidden_layers[-1], 1, bias=False)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self.num_dense_feature], x[:, self.num_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        # 求出线性部分
        linear_logit = [self.embedding_layers_1[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        linear_logit = torch.cat(linear_logit, axis=-1)

        # 求出稀疏特征的embedding向量
        sparse_embeds = [self.embedding_layers_2[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        dnn_input = torch.cat((dense_input, linear_logit, sparse_embeds), dim=-1)

        # DNN 层
        dnn_output = dnn_input
        for dnn in self.dnn_layers:
            dnn_output = dnn(dnn_output)
            dnn_output = torch.tanh(dnn_output)
        dnn_logit = self.dnn_linear(dnn_output)

        # Final
        y_pred = torch.sigmoid(dnn_logit)

        return y_pred