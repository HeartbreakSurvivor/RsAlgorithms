import torch
from FNN.network import FNN
import torch.utils.data as Data
from DeepCrossing.trainer import Trainer
from Utils.criteo_loader import getTestData, getTrainData

fnn_config = \
{
    'embed_dim': 8, # 用于控制稀疏特征经过Embedding层后的稠密特征大小
    'dnn_hidden_units': [128, 128],
    'num_epoch': 150,
    'batch_size': 64,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/FNN.model'
}

if __name__ == "__main__":
    ####################################################################################
    # FNN 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(fnn_config['train_file'], fnn_config['fea_file'])
    train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())

    test_data = getTestData(fnn_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    fnn = FNN(fnn_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)
    print(fnn)
    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=fnn, config=fnn_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    fnn.eval()
    if fnn_config['use_cuda']:
        fnn.loadModel(map_location=lambda storage, loc: storage.cuda(fnn_config['device_id']))
        fnn = fnn.cuda()
    else:
        fnn.loadModel(map_location=torch.device('cpu'))

    y_pred_probs = fnn(torch.tensor(test_data).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("Test Data CTR Predict...\n ", y_pred.view(-1))

