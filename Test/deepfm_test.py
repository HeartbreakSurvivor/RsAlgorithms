import torch
import random
import numpy as np
import pandas as pd
from DeepFM.network import DeepFM
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

deepfm_config = \
    {
        'ebedding_size': 8,
        'dnn_hidden_units': [512,256,128],
        'dnn_dropout': 0.9,
        'num_epoch': 30,
        'batch_size': 32,
        'init_std': 0.001,
        'l2_reg_linear': 1e-3,
        'l2_reg_embedding': 0.00001,
        'lr': 1e-3,
        'l2_regularization': 1e-4,
        'device': 'cpu',
        'seed': 1024,
        'device_id': 0,
        'use_cuda': False,
        'train_file': '../Data/criteo/processed_data/train_set.csv',
        'fea_file': '../Data/criteo/processed_data/fea_col.npy',
        'validate_file': '../Data/criteo/processed_data/val_set.csv',
        'test_file': '../Data/criteo/processed_data/test_set.csv',
        'model_name': '../TrainedModels/DeepFM.model'
    }

if __name__ == "__main__":
    ####################################################################################
    # DeepCrossing 模型
    ####################################################################################
    seed = 1024
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]   #C代表类别特征 class
    dense_features =  ['I' + str(i) for i in range(1, 14)]   #I代表数值特征 int
    col_names = ['id'] + ['label'] + dense_features + sparse_features
    data = pd.read_csv('../Data/criteo/origin_data/train.csv')
    feature_names = sparse_features + dense_features         #全体特征名
    data[sparse_features] = data[sparse_features].fillna('-1', )   # 类别特征缺失 ，使用-1代替
    data[dense_features] = data[dense_features].fillna(0, )        # 数值特征缺失，使用0代替
    target = ['Label']                                             # label

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # 使用LabelEncoder()，为类别特征的每一个item编号
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 数值特征 max-min 0-1归化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # print(data.head(5))

    # 使用一个字典来记录每一个特征的名称，对于稠密特征，类别数设置为1，对于稀疏特征，类别数设置为所有种类出现的次数。
    feat_sizes1={ feat:1 for feat in dense_features}
    feat_sizes2 = {feat: len(data[feat].unique()) for feat in sparse_features}
    feat_sizes={}
    feat_sizes.update(feat_sizes1)
    feat_sizes.update(feat_sizes2)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    # print(train.head(5))
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input =  {name: test[name]  for name in feature_names}

    model = DeepFM(deepfm_config, feat_sizes ,sparse_feature_columns=sparse_features, dense_feature_columns=dense_features)

    model.fit(train_model_input, train[target].values , test_model_input , test[target].values ,batch_size=deepfm_config['batch_size'], epochs=deepfm_config['num_epoch'], verbose=1)

    pred_ans = model.predict(test_model_input, batch_size=deepfm_config['batch_size'])

    print("final test")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))