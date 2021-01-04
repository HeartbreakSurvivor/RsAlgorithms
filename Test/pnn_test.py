import torch
from DeepCross.trainer import Trainer
from PNN.network import PNN
from Utils.criteo_loader import getTestData, getTrainData
import torch.utils.data as Data

pnn_config = \
{
    'L2_dim': 256, # 设置L2隐层的输入维度
    'embed_dim': 8,
    'kernel_type': 'mat',
    'use_inner': False,
    'use_outter': True,
    'num_epoch': 25,
    'batch_size': 32,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/pnn.model'
}

if __name__ == "__main__":
    ####################################################################################
    # PNN 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(pnn_config['train_file'], pnn_config['fea_file'])
    train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())
    test_data = getTestData(pnn_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    pnn = PNN(pnn_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=pnn, config=pnn_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    pnn.eval()
    if pnn_config['use_cuda']:
        pnn.loadModel(map_location=lambda storage, loc: storage.cuda(pnn_config['device_id']))
        pnn = pnn.cuda()
    else:
        pnn.loadModel(map_location=torch.device('cpu'))

    y_pred_probs = pnn(torch.tensor(test_data).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("Test Data CTR Predict...\n ", y_pred.view(-1))

