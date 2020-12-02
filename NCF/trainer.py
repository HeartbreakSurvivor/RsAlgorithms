import torch
from NCF.dataloader import Construct_DataLoader

def pick_optimizer(network, params):
    optimizer = None
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer

class Trainer(object):
    def __init__(self, model, config):
        self._config = config
        self._model = model
        # 选择优化器
        self._optimizer = pick_optimizer(self._model, self._config)
        # 定义损失函数，对于隐反馈数据，这里使用交叉熵损失函数
        self._crit = torch.nn.BCELoss()

    def _train_single_batch(self, users, items, ratings):
        """
        对单个小批量数据进行训练
        :param users: user Tensor
        :param items: item Tensor
        :param ratings: rating Tensor
        :return:
        """
        if self._config['use_cuda'] is True:
            # 将这些数据由CPU迁移到GPU
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        # 先将梯度清零,如果不清零，那么这个梯度就和上一个mini-batch有关
        self._optimizer.zero_grad()
        # 模型的输入users， items，调用forward进行前向传播
        ratings_pred = self._model(users, items)
        # 通过交叉熵损失函数来计算损失, ratings_pred.view(-1)代表将预测结果摊平，变成一维的结构。
        loss = self._crit(ratings_pred.view(-1), ratings)
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降等优化器 更新参数
        self._optimizer.step()
        # 将loss的值提取成python的float类型
        loss = loss.item()
        return loss

    def _train_an_epoch(self, train_loader, epoch_id):
        """
        训练一个Epoch，即将训练集中的所有样本全部都过一遍
        :param train_loader: Torch的DataLoader
        :param epoch_id: 训练轮次Id
        :return:
        """
        # 告诉模型目前处于训练模式，启用dropout以及batch normalization
        self._model.train()
        total_loss = 0
        # 从DataLoader中获取小批量的id以及数据
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            # 这里的user, item, rating大小变成了1024维了，因为batch_size是1024，即每次选取1024个样本数据进行训练
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self._train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        print('Training Epoch: {}, TotalLoss: {}'.format(epoch_id, total_loss))

    def train(self, sampleGenerator):
        # 是否使用GPU加速
        self.use_cuda()
        # 是否使用预先训练好的参数
        self.load_preTrained_weights()

        for epoch in range(self._config['num_epoch']):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            # 每个轮次都重新随机产生样本数据集
            users, items, ratings = sampleGenerator(num_negatives=self._config['num_negative'])
            # 构造一个DataLoader
            data_loader = Construct_DataLoader(users=users, items=items, ratings=ratings,
                                               batchsize=self._config['batch_size'])
            # 训练一个轮次
            self._train_an_epoch(data_loader, epoch_id=epoch)

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def load_preTrained_weights(self):
        if self._config['pretrain'] is True:
            self._model.load_preTrained_weights()

    def save(self):
        self._model.saveModel()
