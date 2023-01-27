
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv2D,Flatten,LSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets= fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.9, shuffle=True, random_state=123
)

print(x_train.shape , x_test.shape) #(18576, 8) (2064, 8)

# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# # x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(18576,4,2)
x_test = x_test.reshape(2064,4,2)

print(x_train.shape , x_test.shape) #(18576, 8) (2064, 8)




# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(100, input_shape=(4,2), return_sequences= True, activation='relu'))
model.add(LSTM(100, activation='relu'))
model.add(Flatten())
model.add(Dense(500, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(500, activation= 'linear'))
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dropout(0.2))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(1, activation= 'linear')) 
model.summary()
# # Total params: 210,801
# # Trainable params: 210,801
# # Non-trainable params: 0

#3.컴파일, 훈련
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor= 'val_loss', 
                             patience= 100, mode = 'min', 
                             restore_best_weights=True, 
                             verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split= 0.3,
                  callbacks=[es], verbose=1)


#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("***********************************")

r2 = r2_score(y_test, y_predict)
print("***********************************")
print("R2 : ", r2)
print("***********************************")


"""
dropout 적용
RMSE :  1.1503178728746182
R2 :  0.05037146751141797

Cnn적용 후 
loss :  [0.686223030090332, 0.6445486545562744]
RMSE :  0.828385796542005
R2 :  0.5075259994669172

LSTM 적용
loss :  [0.5686622858047485, 0.528840959072113]
RMSE :  0.7540969492605148
R2 :  0.5918945567696878

"""