import torch
import numpy as np
from AutoRec.dataloader import Construct_DataLoader

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
        self._model = model
        self._config = config
        self._optimizer = pick_optimizer(self._model, self._config)

    def _train_single_batch(self, batch_x, batch_mask_x):
        """
        对单个小批量数据进行训练
        """
        if self._config['use_cuda'] is True:
            # 将这些数据由CPU迁移到GPU
            batch_x, batch_mask_x = batch_x.cuda(), batch_mask_x.cuda()

        # 模型的输入为用户评分向量或者物品评分向量，调用forward进行前向传播
        ratings_pred = self._model(batch_x.float())
        # 通过交叉熵损失函数来计算损失, ratings_pred.view(-1)代表将预测结果摊平，变成一维的结构。
        loss, rmse = self._model.loss(res=ratings_pred, input=batch_x, mask=batch_mask_x, optimizer=self._optimizer)
        # 先将梯度清零,如果不清零，那么这个梯度就和上一个mini-batch有关
        self._optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降等优化器 更新参数
        self._optimizer.step()
        # 将loss的值提取成python的float类型
        loss = loss.item()
        return loss, rmse

    def _train_an_epoch(self, train_loader, epoch_id, train_mask):
        """
        训练一个Epoch，即将训练集中的所有样本全部都过一遍
        """
        # 告诉模型目前处于训练模式，启用dropout以及batch normalization
        self._model.train()
        total_loss = 0
        total_rmse = 0
        # 从DataLoader中获取小批量的id以及数据
        for batch_id, (batch_x, batch_mask_x) in enumerate(train_loader):
            assert isinstance(batch_x, torch.Tensor)
            assert isinstance(batch_mask_x, torch.Tensor)

            loss, rmse = self._train_single_batch(batch_x, batch_mask_x)
            print('[Training Epoch: {}] Batch: {}, Loss: {}, RMSE: {}'.format(epoch_id, batch_id, loss, rmse))
            total_loss += loss
            total_rmse += rmse
        rmse = np.sqrt(total_rmse.detach().cpu().numpy() / (train_mask == 1).sum())
        print('Training Epoch: {}, Total Loss: {}, total RMSE: {}'.format(epoch_id, total_loss, rmse))

    def train(self, train_r, train_mask_r):
        # 是否使用GPU加速
        self.use_cuda()

        for epoch in range(self._config['num_epoch']):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            # 构造一个DataLoader
            data_loader = Construct_DataLoader(train_r, train_mask_r, batchsize=self._config['batch_size'])
            # 训练一个轮次
            self._train_an_epoch(data_loader, epoch_id=epoch, train_mask=train_mask_r)

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        self._model.saveModel()
