import torch
from AFM.network import AFM
from DeepCrossing.trainer import Trainer
import torch.utils.data as Data
from Utils.criteo_loader import getTestData, getTrainData

afm_config = \
{
    'num_fields': 26, # 这里配置的只是稀疏特征的个数
    'embed_dim': 8, # 用于控制稀疏特征经过Embedding层后的稠密特征大小
    'seed': 1024,
    'l2_reg_w': 0.001,
    'dropout_rate': 0.1,
    'num_epoch': 200,
    'batch_size': 64,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': False,
    'train_file': '../Data/criteo/processed_data/train_set.csv',
    'fea_file': '../Data/criteo/processed_data/fea_col.npy',
    'validate_file': '../Data/criteo/processed_data/val_set.csv',
    'test_file': '../Data/criteo/processed_data/test_set.csv',
    'model_name': '../TrainedModels/AFM.model'
}

if __name__ == "__main__":
    ####################################################################################
    # AFM 模型
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(afm_config['train_file'], afm_config['fea_file'])
    train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())

    test_data = getTestData(afm_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float())

    afm = AFM(afm_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)
    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=afm, config=afm_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    afm.eval()
    if afm_config['use_cuda']:
        afm.loadModel(map_location=lambda storage, loc: storage.cuda(afm_config['device_id']))
        afm = afm.cuda()
    else:
        afm.loadModel(map_location=torch.device('cpu'))

    y_pred_probs = afm(torch.tensor(test_data).float())
    y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print("Test Data CTR Predict...\n ", y_pred.view(-1))

