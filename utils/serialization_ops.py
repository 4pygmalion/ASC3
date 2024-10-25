import pickle


def save_pickle(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)

    return


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
