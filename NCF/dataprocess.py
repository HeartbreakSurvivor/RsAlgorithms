import random
import pandas as pd
import numpy as np
from copy import deepcopy

random.seed(0)

class DataProcess(object):
    def __init__(self, filename):
        self._filename = filename
        self._loadData()
        self._preProcess()
        self._binarize(self._originalRatings)
        # 对'userId'这一列的数据，先去重，然后构成一个用户列表
        self._userPool = set(self._originalRatings['userId'].unique())
        self._itemPool = set(self._originalRatings['itemId'].unique())
        print("user_pool size: ", len(self._userPool))
        print("item_pool size: ", len(self._itemPool))

        self._select_Negatives(self._originalRatings)
        self._split_pool(self._preprocessRatings)

    def _loadData(self):
        self._originalRatings = pd.read_csv(self._filename, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                            engine='python')
        return self._originalRatings

    def _preProcess(self):
        """
        对user和item都重新编号，这里这么做的原因是因为，模型的输入是one-hot向量，需要把user和item都限制在Embedding的长度之内，
        模型的两个输入的长度分别是user和item的数量，所以要重新从0编号。
        """
        # 1. 新建名为"userId"的列，这列对用户从0开始编号
        user_id = self._originalRatings[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id)) #根据user的长度创建一个数组
        # 将原先的DataFrame与user_id按照"uid"这一列进行合并
        self._originalRatings = pd.merge(self._originalRatings, user_id, on=['uid'], how='left')

        # 2. 对物品进行重新排列
        item_id = self._originalRatings[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        self._originalRatings = pd.merge(self._originalRatings, item_id, on=['mid'], how='left')

        # 按照['userId', 'itemId', 'rating', 'timestamp']的顺序重新排列
        self._originalRatings = self._originalRatings[['userId', 'itemId', 'rating', 'timestamp']]
        print(self._originalRatings)
        print('Range of userId is [{}, {}]'.format(self._originalRatings.userId.min(), self._originalRatings.userId.max()))
        print('Range of itemId is [{}, {}]'.format(self._originalRatings.itemId.min(), self._originalRatings.itemId.max()))

    def _binarize(self, ratings):
        """
        binarize data into 0 or 1 for implicit feedback
        """
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        self._preprocessRatings = ratings
        # print("binary: \n", self._preprocessRatings)

    def _select_Negatives(self, ratings):
        """
        Select al;l negative samples and 100 sampled negative items for each user.
        """
        # 构造user-item表
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        print("interact_status: \n", interact_status)

        # 把与用户没有产生过交互的样本都当做是负样本
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self._itemPool - x)

        # 从上面的全部负样本中随机选99个出来
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        print("after sampling interact_status: \n", interact_status)

        print("select and rearrange columns")
        self._negatives = interact_status[['userId', 'negative_items', 'negative_samples']]

    def _split_pool(self, ratings):
        """leave one out train/test split """
        print("sort by timestamp descend")
        # 先按照'userID'进行分组，然后根据时间戳降序排列
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        print(ratings)

        # 选取排名第一的数据作为测试集，也就是最新的那个数据
        test = ratings[ratings['rank_latest'] == 1]
        # 选取所有排名靠后的，也就是历史数据当做训练集
        train = ratings[ratings['rank_latest'] > 1]
        # print("test: \n", test)
        # print("train: \n", train)

        print("size of test {0}, size of train {1}".format(len(test), len(train)))

        # 确保训练集和测试集的userId是一样的
        assert train['userId'].nunique() == test['userId'].nunique()

        self.train_ratings = train[['userId', 'itemId', 'rating']]
        self.test_ratings = test[['userId', 'itemId', 'rating']]

    def sample_generator(self, num_negatives):
        # 合并之后的train_ratings的列包括['userId','itemId'，'rating','negative_items']
        train_ratings = pd.merge(self.train_ratings, self._negatives[['userId', 'negative_items']], on='userId')
        # 从用户的全部负样本集合中随机选择num_negatives个样本当做负样本，并产生一个新的名为"negatives"的列
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        print(train_ratings)

        # 构造模型所需要的数据，分别是输入user、items以及目标分值ratings。
        users, items, ratings = [], [], []
        for row in train_ratings.itertuples():
            # 构造正样本，分别是userId， itemId以及目标分值1
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            # 为每个用户构造num_negatives个负样本，分别是userId， itemId以及目标分值0
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0)) # 负样本的ratings为0，直接强行设置为0

        return users, items, ratings
