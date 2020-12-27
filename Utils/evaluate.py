
def recall(Recommend, Test, N):
    """
    :param Recommend:
    :param Test:
    :param N:
    :return:
    """
    hit = 0
    all = 0
    for user in Test.keys():
        # get the first N recommend items
        ru = Recommend[user][:N]
        tu = Test[user]
        hit += len((ru & tu))
        all += len(tu)
    return hit / (all * 1.0)

def Precision(Recommend, Test, N):
    """
    :param Recommend:
    :param Test:
    :param N:
    :return:
    """
    hit = 0
    all = 0
    for user in Test.keys():
        # get the first N recommend items
        ru = Recommend[user][:N]
        tu = Test[user]
        hit += len((ru & tu))
        all += N
    return hit / (all * 1.0)

def Coverage(Recommend, Test, N):
    recommend_items = set()
    all_items = set()
    for user in Test.keys():
        au = Recommend[user]
        for item in au:
            all_items.add(item)
        ru = Recommend[user][:N]
        for item in ru:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

