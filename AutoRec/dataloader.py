import torch
import numpy as np
import torch.utils.data as Data

def dataProcess(filename, num_users, num_items, train_ratio):
    fp = open(filename, 'r')
    lines = fp.readlines()

    num_total_ratings = len(lines)

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    train_r = np.zeros((num_users, num_items))
    test_r = np.zeros((num_users, num_items))

    train_mask_r = np.zeros((num_users, num_items))
    test_mask_r = np.zeros((num_users, num_items))

    # 生成0~num_total_ratings范围内的的随机序列
    random_perm_idx = np.random.permutation(num_total_ratings)
    # 将数据分为训练集和测试集
    train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_r[user_idx][item_idx] = int(rating)
        train_mask_r[user_idx][item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_r[user_idx][item_idx] = int(rating)
        test_mask_r[user_idx][item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    return train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set

def Construct_DataLoader(train_r, train_mask_r, batchsize):
    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r))
    return Data.DataLoader(dataset=torch_dataset, batch_size=batchsize, shuffle=True)
