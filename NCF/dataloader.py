import torch
from torch.utils.data import DataLoader, Dataset

class UserItemRatingDataset(Dataset):
    """
    Wrapper, convert input <user, item, rating> Tensor into torch Dataset
    """
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self._user_tensor = user_tensor
        self._item_tensor = item_tensor
        self._target_tensor = target_tensor

    def __getitem__(self, index):
        return self._user_tensor[index], self._item_tensor[index], self._target_tensor[index]

    def __len__(self):
        return self._user_tensor.size(0)

def Construct_DataLoader(users, items, ratings, batchsize):
    assert batchsize > 0
    dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                    item_tensor=torch.LongTensor(items),
                                    target_tensor=torch.LongTensor(ratings))
    return DataLoader(dataset, batch_size=batchsize, shuffle=True)