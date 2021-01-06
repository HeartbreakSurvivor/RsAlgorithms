import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel

class BiInteractionPooling(nn.Module):
    """Bi-Interaction Layer used in Neural FM,compress the
      pairwise element-wise product of features into one single vector.
      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.
    """
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

class NFM(BaseModel):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(NFM, self).__init__(config)
        # 稠密和稀疏特征的数量
        self.num_dense_feature = dense_features_cols.__len__()
        self.num_sparse_feature = sparse_features_cols.__len__()

        # NFM的线性部分，对应 ∑WiXi
        self.linear_model = nn.Linear(self.num_dense_feature + self.num_sparse_feature, 1)

        # NFM的Embedding层
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=feat_dim, embedding_dim=config['embed_dim'])
                for feat_dim in sparse_features_cols
        ])

        # B-Interaction 层
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = config['bi_dropout']
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(self.bi_dropout)

        # NFM的DNN部分
        self.hidden_layers = [self.num_dense_feature + config['embed_dim']] + config['dnn_hidden_units']
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
        linear_logit = self.linear_model(x)

        # 求出稀疏特征的embedding向量
        sparse_embeds = [self.embedding_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        # 送入B-Interaction层
        fm_input = sparse_embeds.view(-1, self.num_sparse_feature, self._config['embed_dim'])
        # print(fm_input)
        # print(fm_input.shape)

        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)

        bi_out = bi_out.view(-1, self._config['embed_dim'])
        # 将结果聚合起来
        dnn_input = torch.cat((dense_input, bi_out), dim=-1)

        # DNN 层
        dnn_output = dnn_input
        for dnn in self.dnn_layers:
            dnn_output = dnn(dnn_output)
            # dnn_output = nn.BatchNormalize(dnn_output)
            dnn_output = torch.relu(dnn_output)
        dnn_logit = self.dnn_linear(dnn_output)

        # Final
        logit = linear_logit + dnn_logit
        y_pred = torch.sigmoid(logit)

        return y_pred