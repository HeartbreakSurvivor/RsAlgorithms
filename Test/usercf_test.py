from Utils import movielens_loader
from UserCF import usercf

if __name__ == "__main__":
    ####################################################################################
    # UserCF 基于用户的协同过滤算法
    ####################################################################################
    train, test = movielens_loader.LoadMovieLensData("../Data/ml-1m/ratings.dat", 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))
    UserCF = usercf.UserCF(train)
    UserCF.train()

    # print(UserCF.recommend(list(test.keys())[0], 5, 80))
    # print(UserCF.recommend(list(test.keys())[1], 5, 80))
    # print(UserCF.recommend(list(test.keys())[2], 5, 80))
    # print(UserCF.recommend(list(test.keys())[3], 5, 80))

    # 选取与user最相似的80个用户，并且从这些用户喜爱的物品中选取5个推荐给user
    print("start recommend ...")
    cnt = 0
    for user in test.keys():
        print(UserCF.recommend(user, 5, 80))
        cnt += 1
        if cnt == 5:
            break

    ############################# Toy Example ##############################
    # train = dict({'A':['a','b','d'], 'B':['a','c','d'], 'C':['b','e'], 'D':['c','d','e']})
    # test = dict({'C':['a']})
    # UserCF = usercf.UserCF(train)
    # UserCF.train()
    # print(UserCF.recommend('C', 5, 80))