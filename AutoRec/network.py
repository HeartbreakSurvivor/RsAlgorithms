import torch
import numpy as np
import torch.nn as nn

class AutoRec(nn.Module):
    """
    基于物品的AutoRec模型
    """
    def __init__(self, config):
        super(AutoRec, self).__init__()
        self._num_items = config['num_items']
        self._hidden_units = config['hidden_units']
        self._lambda_value = config['lambda']
        self._config = config

        # 定义编码器结构
        self._encoder = nn.Sequential(
            nn.Linear(self._num_items, self._hidden_units),
            nn.Sigmoid()
        )
        # 定义解码器结构
        self._decoder = nn.Sequential(
            nn.Linear(self._hidden_units, self._num_items)
        )

    def forward(self, input):
        return self._decoder(self._encoder(input))

    def loss(self, res, input, mask, optimizer):
        cost = 0
        temp = 0

        cost += ((res - input) * mask).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            # 找到权重矩阵V和W，并且计算平方和，用于约束项。
            for j in i['params']:
                if j.data.dim() == 2:
                    temp += torch.t(j.data).pow(2).sum()

        cost += temp * self._config['lambda'] * 0.5
        return cost, rmse

    def recommend_user(self, r_u, N):
        """
        :param r_u: 单个用户对所有物品的评分向量
        :param N: 推荐的商品个数
        """
        # 得到用户对所有物品的评分
        predict = self.forward(torch.from_numpy(r_u).float())
        predict = predict.detach().numpy()
        indexs = np.argsort(-predict)[:N]
        return indexs

    def recommend_item(self, user, test_r, N):
        """
        :param r_u: 所有用户对物品i的评分向量
        :param N: 推荐的商品个数
        """
        # 保存给user的推荐列表
        recommends = np.array([])

        for i in range(test_r.shape[1]):
            predict = self.forward(test_r[:, i])
            recommends.append(predict[user])

        # 按照逆序对推荐列表排序，得到最大的N个值的索引
        indexs = np.argsot(-recommends)[:N]
        # 按照用户对物品i的评分降序排序吗，推荐前N个物品给到用户
        return recommends[indexs]

    def evaluate(self, test_r, test_mask_r, user_test_set, user_train_set, item_test_set, item_train_set):
        test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
        test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor)

        res = self.forward(test_r_tensor)

        unseen_user_test_list = list(user_test_set - user_train_set)
        unseen_item_test_list = list(item_test_set - item_train_set)

        for user in unseen_user_test_list:
            for item in unseen_item_test_list:
                if test_mask_r[user, item] == 1:
                    res[user, item] = 3

        mse = ((res - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
        RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
        RMSE = np.sqrt(RMSE)
        print('test RMSE : ', RMSE)

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)