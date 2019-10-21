# 机器学习助手
将机器学习/深度学习的代码构建模块化并提供日志记录 ~~是我在做实验时被记录数据差点搞死之后的结果~~
## 整个流程负责的实体分为三块，模型，数据，训练
## MlModel
可以使用sklearn所提供的模型也可以使用keras模型，其余目前还在开发中。
```python
model = MlModel("LSTM-FC", build_lstm_model((-1, process_args['lag_hour'], len(process_args['feature_cols']))), model_args=None, is_net=True, input_shape=(-1, process_args['lag_hour'], len(process_args['feature_cols'])), task="regression")
```
神经网络的构建不需要提供模型参数，但需要名称，建立好的模型，并指定task是回归还是分类，输入向量的尺寸

### MlDataSet

```python
data = pd.read_csv('./imputated_data.csv')
process_args = dict()
process_args['data'] = data
process_args['split_ratio'] = [0.6, 0.2, 0.2]
process_args['lag_hour'] = 24
process_args['forward_hour'] = 6
process_args['target_name'] = col
dataset = MlDataset(preprocess, process_args)
```
这里其实最为重要的是preprocess的编写，其是对data的处理方法，由使用者自行拟定。所有的参数包括数据都会通过process_args进行传输。并在dataset内部自动分为 train_X, train_y, valid_X, valid_y, test_X, test_y, scaler。使用者在需要指定数据形状变更时，可以通过get_data() 方法进行的第二个参数shape进行指定，也可以直接在preprocess中写入。

### MlTraining

```python
training = MlTraining(model=model, data=dataset, fit_args=network_args)
training.fit()
training.evaluate()
```
MlTraining 实际上只需要模型，数据集和拟合参数三部分，值得注意的时training.evaluate()时才会输出日志。更多功能正在开发中，欢迎issue。

## Finally, having fun with it! Hoping it can be helpful >_< 