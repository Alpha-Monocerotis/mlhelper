# -*- coding: utf-8 -*-
# Developer : Mengyang Liu
# Date		: 2019.10.21
# Filename	: mlhelper.py
# Tool		: Visual Studio Code

from keras.models import Model, Sequential
import pandas as pd
import random
import json
import os
import hashlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import time

# 日志
def savelog(log_dir):
    def log(func):
        def wrapper(*arg, **kw):
            with open(log_dir, 'rb+') as f:
                f.seek(-1,2)
                f.write(b',')
                f.write(bytes(json.dumps(func(*arg, **kw).get('log')).encode('utf-8')))
                f.write(b']')
            return None
        return wrapper
    return log

# 数据集
class MlDataset(object):
    
    
    def __init__(self, process, process_args):
        r"""Here is for overview of the function
        Here is for details
        # Arguments:
              data: 原始数据
              process: 数据处理方法
              process_args: 数据处理方法参数
        # Returns:
              None
        # Raises:
              TypeError : 当data类型不为DataFrame时
        # Tested: False
              Example:
            >>>
        """
        if not isinstance(process_args['data'], pd.DataFrame):
            raise TypeError("data must be a DataFrame")
        hashmd5 = hashlib.md5()
        hashmd5.update(str(time.time()).encode('utf-8'))
        self.did = str(hashmd5.hexdigest())
        self.train_X, self.train_y, self.valid_X, self.valid_y, self.test_X, self.test_y, self.scaler = process(**process_args)

    def get_data(self, label, shape):

        if label == "train":
            return self.train_X.reshape(shape), self.train_y
        elif label == "valid":
            return self.valid_X.reshape(shape), self.valid_y
        else:
            return self.test_X.reshape(shape), self.test_y
    

# 模型
class MlModel(object):
    
    def __init__(self, name, model, model_args=None, is_net=False, input_shape=None, task="regression"):
        # 使用时间戳的MD5加密作为mid
        hashmd5 = hashlib.md5()
        hashmd5.update(str(time.time()).encode('utf-8'))
        self.mid = str(hashmd5.hexdigest())
        self.name = name
        if task not in ["regression", "category"]:
            raise ValueError("task must be either 'regression' or 'category'")
        self.task = task
        if is_net and type(model) not in [Model, Sequential]:
            raise TypeError('model type Error!')
        if not is_net and model_args is None:
            raise ValueError('model_args cannot be None if is_net is False.')
        if model_args is not None and not isinstance(model_args, dict):
            raise TypeError('model_args must be a dict')
        if is_net:
            self.model = model
            self.model_args = None
        else:
            self.model = model(**model_args)
        self.is_net = is_net
        self.input_shape = input_shape
    

    def load_model(self, model, input_shape, model_args=None):
        self.model = model
        self.input_shape = input_shape
        if self.model_args is not None:
            self.model_args = model_args

# 训练
class MlTraining(object):
    
    # 训练实例包含模型，数据集和训练参数三个属性
    def __init__(self, model, data, fit_args):
        hashmd5 = hashlib.md5()
        hashmd5.update(str(time.time()).encode('utf-8'))
        self.tid = str(hashmd5.hexdigest())
        self.model = model
        self.data = data
        self.fit_args = fit_args
        
    # 拟合
    def fit(self):
        if self.model.is_net:
            self.model.model.fit(**self.fit_args)
            self.model.model.save('./models/' + self.model.mid + '.h5')
        else:
            self.model.model.fit(**self.fit_args)
            self.model.model.save('./models/' + self.model.mid + '.pkl')
        print("\033[1;31mTraining of " + self.model.mid + " is complete.\033[0m")
        

    # 模型评估
    @savelog("./log/training.json")
    def evaluate(self):
        if self.model.task == 'regression':
            prediction = self.model.model.predict(self.data.test_X.reshape(self.model.input_shape))

            # 训练评估相关信息
            evaluation_log = dict()
            evaluation_log['mse'] = mean_squared_error(prediction, self.data.test_y)
            evaluation_log['mae'] = mean_absolute_error(prediction, self.data.test_y)
            evaluation_log['r2'] = r2_score(prediction, self.data.test_y)

            # 模型相关信息
            model_log = {}
            model_log['mid'] = self.model.mid
            model_log['model_args'] = self.model.model_args
            model_log['is_net'] = self.model.is_net
            model_log['input_shape'] = self.model.input_shape
            model_log['name'] = self.model.name
            model_log['model_type'] = str(type(self.model))

            # 数据集相关信息
            dataset_log = {}
            dataset_log['did'] = self.data.did
            dataset_log['training_set_shape'] = self.data.train_X.reshape(self.model.input_shape).shape
            dataset_log['valid_set_shape'] = self.data.valid_X.reshape(self.model.input_shape).shape
            dataset_log['test_set_shape'] = self.data.test_X.reshape(self.model.input_shape).shape
            dataset_log['norm'] = True if self.data.scaler is None else False

            
            return {"log": {"tid": self.tid, "evaluation": evaluation_log, "model": model_log, "dataset": dataset_log}}
        print("\033[1;31mEvaluation of " + self.model.mid + " is complete.\033[0m")

    

    

    @savelog("./log/training.json")
    def fit_evaluate(self):
        return {"log":{"mid":self.model.mid, "did":self.data.did}}






if __name__ == "__main__":
    pass



