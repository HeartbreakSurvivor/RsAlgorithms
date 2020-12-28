import torch
from DeepCrossing.trainer import Trainer
from DeepCrossing.network import DeepCrossing
from Utils.criteo_loader import getTestData, getTrainData
import torch.utils.data as Data

deepcrossing_config = \
{
    'embed_dim': 8, # 用于控制稀疏特征经过Embedding层后的稠密特征大小
    'min_dim': 256, # 稀疏特征维度小于min_dim的直接进入stack layer，不用经过embedding层
    'hidden_layers': [512,256,128,64,32],
    'num_epoch': 30,
    'batch_size': 32,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/DeepCrossing.model'
}

if __name__ == "__main__":
    ####################################################################################
    # DeepCrossing 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(deepcrossing_config['train_file'], deepcrossing_config['fea_file'])
    train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())
    test_data = getTestData(deepcrossing_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    deepCrossing = DeepCrossing(deepcrossing_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=deepCrossing, config=deepcrossing_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    deepCrossing.eval()
    if deepcrossing_config['use_cuda']:
        deepCrossing.loadModel(map_location=lambda storage, loc: storage.cuda(deepcrossing_config['device_id']))
        deepCrossing = deepCrossing.cuda()
    else:
        deepCrossing.loadModel(map_location=torch.device('cpu'))

    y_pred_probs = deepCrossing(torch.tensor(test_data).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("Test Data CTR Predict...\n ", y_pred.view(-1))

