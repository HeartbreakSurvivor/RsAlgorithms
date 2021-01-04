import torch
from FM.trainer import Trainer
from FM.network import FM
from Utils.criteo_loader import getTestData, getTrainData
import torch.utils.data as Data

fm_config = \
{
    'latent_dim': 10,
    'num_dense_features': 13, # for criteo data set
    'num_epoch': 10,
    'batch_size': 64,
    'lr': 1e-6,
    'l2_regularization': 1e-3,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/fm.model'
}

def toOneHot(x, MaxList):
    res = []
    for i in range(len(x)):
        t = torch.zeros(MaxList[i])
        t[int(x[i])] = 1
        res.append(t)
    return torch.cat(res, -1)

if __name__ == "__main__":
    ####################################################################################
    # FM 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(fm_config['train_file'], fm_config['fea_file'])
    p = sum(sparse_features_col) + fm_config['num_dense_features']

    rows, cols = training_data.shape
    train_x = torch.zeros((rows, p))
    for row in range(rows):
        dense = torch.Tensor(training_data[row][:fm_config['num_dense_features']])
        sparse = training_data[row][fm_config['num_dense_features']:]
        sparse = toOneHot(sparse, sparse_features_col)
        train_x[row] = torch.cat((dense, sparse),0)

    train_dataset = Data.TensorDataset(train_x.float().clone().detach().requires_grad_(True), torch.tensor(training_label).float())
    test_data = getTestData(fm_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    fm = FM(fm_config, p)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=fm, config=fm_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

