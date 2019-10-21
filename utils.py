import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
class Scaler(object):

    def __init__(self):
        self.min = 0
        self.max = 1
    
    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)
    
    def transform(self, X):
        for i in range(len(X)):
            X[i] = (X[i] - self.min) / (self.max - self.min)
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    # Transform time-series data into supervised learning data.
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def preprocess(**args):

    # 获取数据处理方法基础变量
    data = args.get('data')
    split_ratio = args.get('split_ratio')
    scaler = args.get('normalize_scaler')
    block_size = args.get('block_size')
    feature_cols = args.get('feature_cols')
    lag_hour = args.get('lag_hour')
    forward_hour = args.get('forward_hour')
    target_index = args.get('target_index')

    # 检查split_ratio类型,
    try:
        if len(split_ratio) != 3:
            raise ValueError('Split ratio must be a three-digit sequence.')
        elif sum(split_ratio) != 1:
            raise ValueError("Split_ratio's sum must be 1.")
    except Exception:
        raise RuntimeError("Seems that your split_ratio is a suitable type.")

    # 去除无意义变量
    data = data[feature_cols]
    train_vali_set = None
    test_set = None

    # 对数据进行分块抽取处理
    for i in range(block_size):
        if train_vali_set is None:
            train_vali_set = data.values[i * int(len(data) / block_size): int((i + sum(split_ratio[0:2])) * len(data) / block_size)]
        else:
            train_vali_set = np.vstack((train_vali_set, data.values[i * int(len(data) / block_size): int((i + sum(split_ratio[0:2])) * len(data) / block_size)]))
        if test_set is None:
            test_set = data.values[int((i + sum(split_ratio[0:2])) * len(data) / block_size) : (i + 1) * int(len(data) / block_size)]
        else:
            test_set = np.vstack((test_set, data.values[int((i + sum(split_ratio[0:2])) * len(data) / block_size) : (i + 1) * int(len(data)/ block_size)]))
    
    # 归一化处理
    scaler.fit(train_vali_set)
    train_vali_set = scaler.transform(train_vali_set)
    test_set = scaler.transform(test_set)

    # 序列数据转化为监督学习数据
    supervised_train_vali_set = series_to_supervised(train_vali_set, lag_hour, forward_hour, True).values
    supervised_test_set = series_to_supervised(test_set, lag_hour, forward_hour, True).values
    supervised_train_vali_set_X = supervised_train_vali_set[:, 0:lag_hour * len(feature_cols)]
    supervised_train_vali_set_y = supervised_train_vali_set[:, (lag_hour + forward_hour - 1) + target_index]
    # X 和 y的分开，送入train_test_split进行随机打乱并创建训练集和验证集
    train_X, valid_X, train_y, valid_y = train_test_split(supervised_train_vali_set_X, supervised_train_vali_set_y, random_state=10)
    test_X = supervised_test_set[:, 0:lag_hour * len(feature_cols)]
    test_y = supervised_test_set[:, (lag_hour + forward_hour - 1) + target_index]
    return train_X, train_y, valid_X, valid_y, test_X, test_y, scaler

