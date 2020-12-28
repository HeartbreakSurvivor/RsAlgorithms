import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)

class Trainer(object):
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config['lr'], weight_decay=config['l2_regularization'])
        self._loss_func = torch.nn.BCELoss()

    def _train_single_batch(self, x, labels):
        """
        对单个小批量数据进行训练
        """
        self._optimizer.zero_grad()
        y_predict = self._model(x)
        loss = self._loss_func(y_predict.view(-1), labels)

        loss.backward()
        self._optimizer.step()

        loss = loss.item()
        return loss, y_predict

    def _train_an_epoch(self, train_loader, epoch_id):
        """
        训练一个Epoch，即将训练集中的所有样本全部都过一遍
        """
        # 设置模型为训练模式，启用dropout以及batch normalization
        self._model.train()

        total = 0
        # 从DataLoader中获取小批量的id以及数据
        for batch_id, (x, labels) in enumerate(train_loader):
            x = Variable(x)
            labels = Variable(labels)
            if self._config['use_cuda'] is True:
                x, labels = x.cuda(), labels.cuda()

            loss, predicted = self._train_single_batch(x, labels)

            total += loss
            print('[Training Epoch: {}] Batch: {}, Loss: {}'.format(epoch_id, batch_id, loss))
        print("Training Epoch: %d, total loss: %f" % (epoch_id, total))

    def train(self, train_dataset):
        self.use_cuda()
        for epoch in range(self._config['num_epoch']):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            # 构造DataLoader
            data_loader = DataLoader(dataset=train_dataset, batch_size=self._config['batch_size'], shuffle=True)
            # 训练一个轮次
            self._train_an_epoch(data_loader, epoch_id=epoch)

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        self._model.saveModel()
