from Utils import movielens_loader
from LFM import lfm

if __name__ == "__main__":
    ####################################################################################
    # LFM 隐语义模型算法
    ####################################################################################
    train, test = movielens_loader.LoadMovieLensData("../Data/ml-1m/ratings.dat", 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))
    lfm = lfm.LFM(train, 0.02, 0.01, 10, 30)
    lfm.train()
    print(lfm.user_p)
    print(lfm.item_q)

    # 给测试集中的用户推荐5部电影
    print("start recommend ...")
    cnt = 0
    for user in test.keys():
        print(lfm.recommend(user, 5))
        cnt += 1
        if cnt == 5:
            break
