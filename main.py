from mlhelper import *
import pandas as pd
from utils import preprocess
from sklearn.preprocessing import MinMaxScaler
from model import build_lstm_model



if __name__ == '__main__':

   data = pd.read_csv('./imputated_data.csv')
   process_args = dict()
   process_args['data'] = data
   process_args['split_ratio'] = [0.6, 0.2, 0.2]
   process_args['normalize_scaler'] = MinMaxScaler()
   process_args['block_size'] = int(43800 / 219)
   process_args['feature_cols'] = ['AQI', 'PM2_5', 'PM2_5_24H', 'PM_10', 'PM_10_24H', 'SO2','SO2_24H', 'NO2', 'NO2_24H', 'O3', 'O3_24H', 'O3_8H', 'O3_8H_24H', 'CO','CO_24H']
   for col in ['PM2_5', 'PM_10', 'SO2', 'NO2', 'O3', 'CO']:
      process_args['target_index'] = process_args['feature_cols'].index(col)
      process_args['lag_hour'] = 24
      process_args['forward_hour'] = 6
      process_args['target_name'] = col
      dataset = MlDataset(preprocess, process_args)

      model = MlModel("LSTM-FC", build_lstm_model((-1, process_args['lag_hour'], len(process_args['feature_cols']))), model_args=None, is_net=True, input_shape=(-1, process_args['lag_hour'], len(process_args['feature_cols'])), task="regression")
      network_args = dict()
      network_args['epochs'] = 200
      network_args['batch_size'] = 256
      network_args['shuffle'] = False
      network_args['x'] = dataset.get_data(label='train', shape=model.input_shape)[0]
      network_args['y'] = dataset.get_data(label='train', shape=model.input_shape)[1]
      network_args['validation_data'] = (dataset.get_data(label='valid', shape=model.input_shape))

      training = MlTraining(model=model, data=dataset, fit_args=network_args)
      training.fit()
      training.evaluate()
