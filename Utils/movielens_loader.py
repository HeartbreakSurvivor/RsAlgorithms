import random
import pandas as pd

def LoadMovieLensData(filepath, train_rate):
    ratings = pd.read_table(filepath, sep="::", header=None, names=["UserID", "MovieID", "Rating", "TimeStamp"],\
                            engine='python')
    ratings = ratings[['UserID','MovieID']]

    train = []
    test = []
    random.seed(3)
    for idx, row in ratings.iterrows():
        user = int(row['UserID'])
        item = int(row['MovieID'])
        if random.random() < train_rate:
            train.append([user, item])
        else:
            test.append([user, item])
    return PreProcessData(train), PreProcessData(test)

def PreProcessData(originData):
    """
    建立User-Item表，结构如下：
        {"User1": {MovieID1, MoveID2, MoveID3,...}
         "User2": {MovieID12, MoveID5, MoveID8,...}
         ...
        }
    """
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set())
        trainData[user].add(item)
    return trainData
