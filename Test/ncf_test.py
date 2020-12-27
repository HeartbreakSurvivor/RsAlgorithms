from NCF.dataprocess import DataProcess
from NCF.network import GMF,MLP,NeuMF
from NCF.trainer import Trainer

import torch

gmf_config = {'num_epoch': 1,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim_gmf': 8,
              'num_negative': 4,
              'layers': [],
              'l2_regularization': 0, # 0.01
              'pretrain': False, # do not modify this
              'use_cuda': False,
              'device_id': 2,
              'model_name': '../TrainedModels/NCF_GMF.model'
              }

mlp_config = {'num_epoch': 1,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim_mlp': 8,
              'latent_dim_gmf': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': 2,
              'pretrain': True,
              'gmf_config': gmf_config,
              'pretrain_gmf': '../TrainedModels/NCF_GMF.model',
              'model_name': '../TrainedModels/NCF_MLP.model'
              }

neumf_config = {'num_epoch': 1,
                'batch_size': 128, #1024
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_gmf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'alpha': 0.5, # 用于控制GMF和MLP模型参数的权重
                'use_cuda': False,
                'device_id': 2,
                'pretrain': False, # if you set this to True, you must guarantee the  Neumf layers is the same as the mlp layers
                'gmf_config': gmf_config,
                'pretrain_gmf': '../TrainedModels/NCF_GMF.model',
                'mlp_config': mlp_config,
                'pretrain_mlp': '../TrainedModels/NCF_MLP.model',
                'model_name': '../TrainedModels/NCF_NeuMF.model'
                }

if __name__ == "__main__":
    ####################################################################################
    # NCF 神经协同过滤算法
    ####################################################################################

    # 加载和预处理数据
    dp = DataProcess("../Data/ml-1m/ratings.dat")

    # 初始化GMP模型
    # config = gmf_config
    # model = GMF(config, config['latent_dim_gmf'])

    # # 初始化MLP模型
    # config = mlp_config
    # model = MLP(config, config['latent_dim_mlp'])

    # 初始化NeuMF模型
    config = neumf_config
    model = NeuMF(config, config['latent_dim_gmf'], config['latent_dim_mlp'])

    # ###############################################################
    # 模型训练阶段
    # ###############################################################
    trainer = Trainer(model=model, config=config)
    trainer.train(dp.sample_generator)
    trainer.save()

    # ###############################################################
    # 模型测试阶段
    # ###############################################################

    # 加载数据集
    dp = DataProcess("../Data/ml-1m/ratings.dat")

    config = neumf_config
    neumf = NeuMF(config, config['latent_dim_gmf'], config['latent_dim_mlp'])
    state_dict = torch.load("../TrainedModels/NCF_NeuMF.model", map_location=torch.device('cpu'))
    neumf.load_state_dict(state_dict, strict=False)

    print(neumf.forward(torch.LongTensor([1]), torch.LongTensor([1193])))
    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([661])))
    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([914])))
    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([3408])))

    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([1245])))
    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([32])))
    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([4])))
    print(neumf.forward(torch.LongTensor([1]),torch.LongTensor([62])))
