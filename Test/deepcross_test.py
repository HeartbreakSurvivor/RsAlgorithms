import torch
from DeepCross.trainer import Trainer
from DeepCross.network import DeepCross
from Utils.criteo_loader import getTestData, getTrainData
import torch.utils.data as Data

deepcross_config = \
{
    'deep_layers': [256,128,64,32], # 设置Deep模块的隐层大小
    'num_cross_layers': 4, # cross模块的层数
    'num_epoch': 2,
    'batch_size': 32,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/DeepCross.model'
}

if __name__ == "__main__":
    ####################################################################################
    # DeepCross 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(deepcross_config['train_file'], deepcross_config['fea_file'])
    train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())
    test_data = getTestData(deepcross_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    deepCross = DeepCross(deepcross_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=deepCross, config=deepcross_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    deepCross.eval()
    if deepcross_config['use_cuda']:
        deepCross.loadModel(map_location=lambda storage, loc: storage.cuda(deepcross_config['device_id']))
        deepCross = deepCross.cuda()
    else:
        deepCross.loadModel(map_location=torch.device('cpu'))

    y_pred_probs = deepCross(torch.tensor(test_data).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("Test Data CTR Predict...\n ", y_pred.view(-1))

