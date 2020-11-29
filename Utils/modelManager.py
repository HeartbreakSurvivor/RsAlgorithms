import os
import pickle

def save(filename, *data):
    with open(filename, "wb") as f:
        for d in data:
            pickle.dump(d, f)

def load(filename, num=1):
    with open(filename, "rb") as f:
        ret = []
        for i in range(num):
            data = pickle.load(f)
            ret.append(data)
    return ret