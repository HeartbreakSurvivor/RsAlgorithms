import random
import numpy as np
from operator import itemgetter
from Utils import modelManager

class LFM(object):
    def __init__(self, trainData, alpha, regularization_rate, number_LatentFactors=10, number_epochs=10):
        self._trainData = trainData # User-Item表
        self._alpha = alpha # 学习率
        self._lmbda = regularization_rate # 正则化惩罚因子
        self._k = number_LatentFactors # 隐语义类别数量
        self._epochs = number_epochs # 训练次数
        self._item_pool = self._getAllItems() # 所有物品集合
        self._init_matrix()

    def _getAllItems(self):
        # 获取全体物品列表
        print("start collect all items...")
        items_pool = set()
        for user, items in self._trainData.items():
            for item in items:
                items_pool.add(item)
        return list(items_pool)

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时选择高斯分布的随机值作为初始值
        '''
        print("start build latent matrix.")
        # User-LF user_p 是一个 m x k 的矩阵,其中m是用户的数量，k是隐含类别的数量
        self.user_p = dict()
        for user in self._trainData.keys():
            self.user_p[user] = np.random.normal(size=(self._k))
        # Item-LF movie_q 是一个 n x k 的矩阵,其中n是电影的数量，k是隐含类别的数量
        self.item_q = dict()
        for movie in self._item_pool:
            self.item_q[movie] = np.random.normal(size=(self._k))

    def predict(self, user, item):
        # 通过公式 Rui = ∑P(u,k)Q(k,i)求出user对item的感兴趣程度
        return np.dot(self.user_p[user], self.item_q[item])

    def _select_negatives(self, movies):
        """
        选择负样本
        :param movies: 一个用户喜爱的电影集合
        :return: 包含正负样本的电影样本集合
        """
        ret = dict()
        for i in movies: # 记录正样本，兴趣度为1
            ret[i] = 1

        number = 0
        while number < len(movies):
            # 从所有商品集合中随机选取一个当做负样本，兴趣度置为0
            negative_sample = random.choice(self._item_pool)
            if negative_sample in ret:
                continue
            ret[negative_sample] = 0
            number += 1
        return ret

    def _loss(self):
        C = 0.
        for user, user_latent in self.user_p.items():
            for movie, movie_latent in self.item_q.items():
                # try:
                #     rui = self._trainData[user][movie]
                # except KeyError:
                #     rui = 0
                rui = 0
                for u, m in self._trainData.items():
                    if user == u:
                        if movie in m: # 如果movie出现在了user的喜爱列表里面，则rui=1
                            rui = 1
                        break
                    else:
                        continue

                eui = rui - self.predict(user, movie)
                C += (np.square(eui) +
                      self._lmbda * np.sum(np.square(self.user_p[user])) +
                      self._lmbda * np.sum(np.square(self.item_q[movie])))
        return C

    def SGD(self):
        # 随机梯度下降算法
        alpha = self._alpha
        for epoch in range(self._epochs):
            print("############ start {0}th epoch training ##########".format(epoch))
            for user, positive_movies in self._trainData.items():
                # 每次迭代都对用户重新选择负样本
                select_samples = self._select_negatives(positive_movies)
                for movie, rui in select_samples.items():
                    # 使用模型去预测user对movie的相似度，并且得到与真实值之间的误差
                    eui = rui - self.predict(user, movie)
                    # print("current error : ", eui)
                    user_latent = self.user_p[user]
                    movie_latent = self.item_q[movie]
                    # 更新参数
                    self.user_p[user] += alpha * (eui * movie_latent - self._lmbda * user_latent)
                    self.item_q[movie] += alpha * (eui * user_latent - self._lmbda * movie_latent)
            alpha *= 0.9
            print("######### {}td training finished, loss: {} ##########".format(epoch, self._loss()))

    def recommend(self, user, N):
        """
        给user推荐N个商品
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()

        for movie in self._item_pool:
            recommends[movie] = self.predict(user, movie)

        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def train(self):
        try:
            print("start load latent factor matrix P and Q")
            model = modelManager.load("../TrainedModels/lfm.pkl", 3)
            self.user_p = model[0]
            self.item_q = model[1]
            self._item_pool = model[2]
        except BaseException as e:
            print("Exception occurs: " + str(e))
            print("load latent factor matrix failed, start train...")
            self.SGD()
            modelManager.save("../TrainedModels/lfm.pkl", self.user_p, self.item_q, self._item_pool)