from Utils import movielens_loader
from ItemCF import itemcf

if __name__ == "__main__":
    ####################################################################################
    # ItemCF 基于物品的协同过滤算法
    ####################################################################################

    train, test = movielens_loader.LoadMovieLensData("../Data/ml-1m/ratings.dat", 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))
    ItemCF = itemcf.ItemCF(train, similarity='iuf', norm=True)
    ItemCF.train()

    print("start recommend ...")
    cnt = 0
    for user in test.keys():
        print(ItemCF.recommend(user, 5, 80))
        cnt += 1
        if cnt == 5:
            break

    ############################# Toy Example ##############################
    # train = dict({'A':['a','b','d'], 'B':['b','c','e'], 'C':['c','d'], 'D':['b','c','d'], 'E':['a','d']})
    # test = dict({'C':['a']})
    # ItemCF = itemcf.ItemCF(train, similarity='iuf', norm=True)
    # ItemCF.train()
    #
    # print(ItemCF.recommend('C', 5, 80))
