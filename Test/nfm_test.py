import torch
from DeepCrossing.trainer import Trainer
from NFM.network import NFM
import torch.utils.data as Data
from Utils.criteo_loader import getTestData, getTrainData

nfm_config = \
{
    'embed_dim': 8, # 用于控制稀疏特征经过Embedding层后的稠密特征大小
    'dnn_hidden_units': [128, 128],
    'num_dense_features': 13,
    'bi_dropout': 0.5,
    'num_epoch': 500,
    'batch_size': 128,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/NFM.model'
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
    # NFM 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(nfm_config['train_file'], nfm_config['fea_file'])
    train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())

    test_data = getTestData(nfm_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    nfm = NFM(nfm_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)
    print(nfm)
    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=nfm, config=nfm_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    nfm.eval()
    if nfm_config['use_cuda']:
        nfm.loadModel(map_location=lambda storage, loc: storage.cuda(nfm_config['device_id']))
        nfm = nfm.cuda()
    else:
        nfm.loadModel(map_location=torch.device('cpu'))

    y_pred_probs = nfm(torch.tensor(test_data).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("Test Data CTR Predict...\n ", y_pred.view(-1))

