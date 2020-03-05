import numpy as np

def read_from_txt(fpath):
    # load in the file path
    X = []
    y = []
    with open(fpath, 'r') as fin:
        for i in range(6):
            fin.readline()
        for line in fin:
            line = list(map(int, line.strip().split(' ')))
            X.append(line[:-1])
            y.append(line[-1])
    X = np.array(X)
    y = np.array(y)

    # make labels index at 0 starting
    y -= 1
    return X, y