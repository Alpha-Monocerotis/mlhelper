from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Activation
from keras import regularizers


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=(input_shape[1], input_shape[
        2]), return_sequences=True))
    # model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.25))
    # model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    # model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    # model.add(Dropout(0.25))
    
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01),activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))  # 定义输出层神经元个数为1个，即输出只有1维
    model.add(Activation('elu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()
    # model.add(Activation('sigmoid'))#根据情况添加激活函数
    return model