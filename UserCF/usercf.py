import math
from Utils import modelManager
from collections import defaultdict
from operator import itemgetter

class UserCF(object):
    """ User based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine"):
        self._trainData = trainData
        self._similarity = similarity
        self._userSimMatrix = dict() # 用户相似度矩阵

    def similarity(self):
        # 建立User-Item倒排表
        item_user = dict()
        for user, items in self._trainData.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        # 建立用户物品交集矩阵W, 其中C[u][v]代表的含义是用户u和用户v之间共同喜欢的物品数
        for item, users in item_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self._userSimMatrix.setdefault(u, defaultdict(int))
                    if self._similarity == "cosine":
                        self._userSimMatrix[u][v] += 1 #将用户u和用户v共同喜欢的物品数量加一
                    elif self._similarity == "iif":
                        self._userSimMatrix[u][v] += 1. / math.log(1 + len(users))

        # 建立用户相似度矩阵
        for u, related_user in self._userSimMatrix.items():
            # 相似度公式为 |N[u]∩N[v]|/sqrt(N[u]||N[v])
            for v, cuv in related_user.items():
                nu = len(self._trainData[u])
                nv = len(self._trainData[v])
                self._userSimMatrix[u][v] = cuv / math.sqrt(nu * nv)

    def recommend(self, user, N, K):
        """
        用户u对物品i的感兴趣程度：
            p(u,i) = ∑WuvRvi
            其中Wuv代表的是u和v之间的相似度， Rvi代表的是用户v对物品i的感兴趣程度，因为采用单一行为的隐反馈数据，所以Rvi=1。
            所以这个表达式的含义是，要计算用户u对物品i的感兴趣程度，则要找到与用户u最相似的K个用户，对于这k个用户喜欢的物品且用户u
            没有反馈的物品，都累加用户u与用户v之间的相似度。
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        # 先获取user具有正反馈的item数组
        related_items = self._trainData[user]
        # 将其他用户与user按照相似度逆序排序之后取前K个
        for v, sim in sorted(self._userSimMatrix[user].items(), key=itemgetter(1), reverse=True)[:K]:
            # 从与user相似的用户的喜爱列表中寻找可能的物品进行推荐
            for item in self._trainData[v]:
                # 如果与user相似的用户喜爱的物品与user喜欢的物品重复了，直接跳过
                if item in related_items:
                    continue
                recommends.setdefault(item, 0.)
                recommends[item] += sim
        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def train(self):
        try:
            print("start load user similarity matrix")
            self._userSimMatrix = modelManager.load("../TrainedModels/usercf.pkl")[0]
        except BaseException as e:
            print("Exception occurs: " + str(e))
            print("load user similarity matrix failed, start train...")
            self.similarity()
            # save user similarity matrix
            modelManager.save("../TrainedModels/usercf.pkl", self._userSimMatrix)

