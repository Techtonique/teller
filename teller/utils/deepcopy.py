import pickle


def deepcopy(x):

    return pickle.loads(pickle.dumps(x, -1))
