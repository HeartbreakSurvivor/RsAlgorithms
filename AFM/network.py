import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class AFM(BaseModel):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(AFM, self).__init__(config)
        self.num_fields = config['num_fields']
        self.embed_dim = config['embed_dim']
        self.l2_reg_w = config['l2_reg_w']

        # 稠密和稀疏特征的数量
        self.num_dense_feature = dense_features_cols.__len__()
        self.num_sparse_feature = sparse_features_cols.__len__()

        # AFM的线性部分，对应 ∑W_i*X_i, 这里包含了稠密和稀疏特征
        self.linear_model = nn.Linear(self.num_dense_feature + self.num_sparse_feature, 1)

        # AFM的Embedding层,只是针对稀疏特征，有待改进。
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=feat_dim, embedding_dim=config['embed_dim'])
                for feat_dim in sparse_features_cols
        ])

        # Attention Network
        self.attention = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.projection = torch.nn.Linear(self.embed_dim, 1, bias=False)
        self.attention_dropout = nn.Dropout(config['dropout_rate'])

        # prediction layer
        self.predict_layer = torch.nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        # 先区分出稀疏特征和稠密特征，这里是按照列来划分的，即所有的行都要进行筛选
        dense_input, sparse_inputs = x[:, :self.num_dense_feature], x[:, self.num_dense_feature:]
        sparse_inputs = sparse_inputs.long()

        # 求出线性部分
        linear_logit = self.linear_model(x)

        # 求出稀疏特征的embedding向量
        sparse_embeds = [self.embedding_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        sparse_embeds = sparse_embeds.view(-1, self.num_sparse_feature, self.embed_dim)

        # calculate inner product
        row, col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(i), col.append(j)
        p, q = sparse_embeds[:, row], sparse_embeds[:, col]
        inner_product = p * q

        # 通过Attention network得到注意力分数
        attention_scores = torch.relu(self.attention(inner_product))
        attention_scores = torch.softmax(self.projection(attention_scores), dim=1)

        # dim=1 按行求和
        attention_output = torch.sum(attention_scores * inner_product, dim=1)
        attention_output = self.attention_dropout(attention_output)

        # Prodict Layer
        # for regression problem with MSELoss
        y_pred = self.predict_layer(attention_output) + linear_logit
        # for classifier problem with LogLoss
        y_pred = torch.sigmoid(y_pred)
        return y_pred