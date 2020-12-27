import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class NCF(ABC):
    def __init__(self, config, latent_dim_gmf=8, latent_dim_mlp=8):
        self._config = config
        self._num_users = config['num_users']
        self._num_items = config['num_items']
        self._latent_dim_gmf = latent_dim_gmf
        self._latent_dim_mlp = latent_dim_mlp

        # 建立MLP模型的user Embedding层和item Embedding层，输入的向量长度分别为用户的数量，item的数量，输出都是隐式空间的维度latent dim
        self._embedding_user_mlp = torch.nn.Embedding(num_embeddings=self._num_users, embedding_dim=self._latent_dim_mlp)
        self._embedding_item_mlp = torch.nn.Embedding(num_embeddings=self._num_users, embedding_dim=self._latent_dim_mlp)
        # 建立GMP模型的user Embedding层和item Embedding层，输入的向量长度分别为用户的数量，item的数量，输出都是隐式空间的维度latent dim
        self._embedding_user_gmf = torch.nn.Embedding(num_embeddings=self._num_users, embedding_dim=self._latent_dim_gmf)
        self._embedding_item_gmf = torch.nn.Embedding(num_embeddings=self._num_users, embedding_dim=self._latent_dim_gmf)

        # 全连接层
        self._fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self._fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 激活函数
        self._logistic = nn.Sigmoid()

    @property
    def fc_layers(self):
        return self._fc_layers

    @property
    def embedding_user_gmf(self):
        return self._embedding_user_gmf

    @property
    def embedding_item_gmf(self):
        return self._embedding_item_gmf

    @property
    def embedding_user_mlp(self):
        return self._embedding_user_mlp

    @property
    def embedding_item_mlp(self):
        return self._embedding_item_mlp

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    @abstractmethod
    def load_preTrained_weights(self):
        pass

class GMF(NCF, nn.Module):
    def __init__(self, config, latent_dim_gmf):
        nn.Module.__init__(self)
        NCF.__init__(self, config=config, latent_dim_gmf=latent_dim_gmf)
        # 创建一个线性模型，输入为潜在特征向量，输出向量长度为1
        self._affine_output = nn.Linear(in_features=self._latent_dim_gmf, out_features=1)

    @property
    def affine_output(self):
        return self._affine_output

    def forward(self, user_indices, item_indices):
        """
        前向传播
        :param user_indices: user Tensor
        :param item_indices: item Tensor
        :return: predicted rating
        """
        # 先将user和item转换为对应的Embedding表示，注意这个支持Tensor操作，即传入的是一个user列表，对其中每一个user都会执行Embedding操作，即都会使用Embedding表示
        user_embedding = self._embedding_user_gmf(user_indices)
        item_embedding = self._embedding_item_gmf(item_indices)
        # 对user_embedding和user_embedding进行逐元素相乘, 这一步其实就是MF算法的实现
        element_product = torch.mul(user_embedding, item_embedding)
        # 将逐元素的乘积的结果通过一个S型神经元
        logits = self._affine_output(element_product)
        rating = self._logistic(logits)
        return rating

    def load_preTrained_weights(self):
        pass

class MLP(NCF, nn.Module):
    def __init__(self, config, latent_dim_mlp):
        nn.Module.__init__(self)
        NCF.__init__(self, config=config, latent_dim_mlp=latent_dim_mlp)
        # 创建一个线性模型，输入为潜在特征向量，输出向量长度为1
        self._affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)

    @property
    def affine_output(self):
        return self._affine_output

    def forward(self, user_indices, item_indices):
        """
        :param user_indices: user Tensor
        :param item_indices: item Tensor
        """
        # 先将user和item转换为对应的Embedding表示，注意这个支持Tensor操作，即传入的是一个user列表，
        # 对其中每一个user都会执行Embedding操作，即都会使用Embedding表示
        user_embedding = self._embedding_user_mlp(user_indices)
        item_embedding = self._embedding_item_mlp(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1) # concat latent vector
        for idx, _ in enumerate(range(len(self._fc_layers))):
            vector = self._fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            ##  Batch normalization
            # vector = torch.nn.BatchNorm1d()(vector)
            ## DroupOut layer
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self._affine_output(vector)
        rating = self._logistic(logits)
        return rating

    def load_preTrained_weights(self):
        config = self._config
        gmf_model = GMF(config, config['latent_dim_gmf'])
        if config['use_cuda'] is True:
            gmf_model.cuda()
        # 加载GMF模型参数到指定的GPU上
        state_dict = torch.load(self._config['pretrain_gmf'])
                                #map_location=lambda storage, loc: storage.cuda(device=self._config['device_id']))
                                #map_location = {'cuda:0': 'cpu'})
        gmf_model.load_state_dict(state_dict, strict=False)

        self._embedding_item_mlp.weight.data = gmf_model.embedding_item_gmf.weight.data
        self._embedding_user_mlp.weight.data = gmf_model.embedding_user_gmf.weight.data

class NeuMF(NCF, nn.Module):
    def __init__(self, config, latent_dim_gmf, latent_dim_mlp):
        nn.Module.__init__(self)
        NCF.__init__(self, config, latent_dim_gmf, latent_dim_mlp)

        # 创建一个线性模型，输入为GMF模型和MLP模型的潜在特征向量长度之和，输出向量长度为1
        self._affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_gmf'], out_features=1)

    @property
    def affine_output(self):
        return self._affine_output

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self._embedding_user_mlp(user_indices)
        item_embedding_mlp = self._embedding_item_mlp(item_indices)
        user_embedding_gmf = self._embedding_user_gmf(user_indices)
        item_embedding_gmf = self._embedding_item_gmf(item_indices)

        # concat the two latent vector
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        # multiply the two latent vector
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

        for idx, _ in enumerate(range(len(self._fc_layers))):
            mlp_vector = self._fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, gmf_vector], dim=-1)
        logits = self._affine_output(vector)
        rating = self._logistic(logits)
        return rating

    def load_preTrained_weights(self):
        # 加载MLP模型参数
        mlp_model = MLP(self._config['mlp_config'], self._config['mlp_config']['latent_dim_mlp'])
        if self._config['use_cuda'] is True:
            mlp_model.cuda()
        state_dict = torch.load(self._config['pretrain_mlp'])
                                # map_location=lambda storage, loc: storage.cuda(device=self._config['device_id']))
                                # map_location = {'cuda:0': 'cpu'})
        mlp_model.load_state_dict(state_dict, strict=False)

        self._embedding_item_mlp.weight.data = mlp_model.embedding_item_mlp.weight.data
        self._embedding_user_mlp.weight.data = mlp_model.embedding_user_mlp.weight.data
        for idx in range(len(self._fc_layers)):
            self._fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        # 加载GMF模型参数
        gmf_model = GMF(self._config['gmf_config'], self._config['gmf_config']['latent_dim_gmf'])
        if self._config['use_cuda'] is True:
            gmf_model.cuda()
        state_dict = torch.load(self._config['pretrain_gmf'])
                                # map_location=lambda storage, loc: storage.cuda(device=self._config['device_id']))
                                # map_location = {'cuda:0': 'cpu'})
        mlp_model.load_state_dict(state_dict, strict=False)

        self._embedding_item_gmf.weight.data = gmf_model.embedding_item_gmf.weight.data
        self._embedding_user_gmf.weight.data = gmf_model.embedding_user_gmf.weight.data

        self._affine_output.weight.data = self._config['alpha'] * torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self._affine_output.bias.data = self._config['alpha'] * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)
