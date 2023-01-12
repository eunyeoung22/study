import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

path ='./_save/'
# path = '../_save/'        path = 'c:/study/_save/'
# model.save('./_save/keras29_1_save_model.h5')
# model.save(path + 'keras29_1_save_model.h5')


#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, 
    random_state=123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성(함수형)
input1 = Input(shape=(13,))
dence1 = Dense(100, activation= 'linear')(input1)
dence2 = Dense(100, activation= 'sigmoid')(dence1)
dence3 = Dense(200, activation= 'linear')(dence2)
dence4 = Dense(300, activation= 'linear')(dence3)
dence5 = Dense(400, activation= 'linear')(dence4)
dence6 = Dense(500, activation= 'linear')(dence5)
dence7 = Dense(400, activation= 'relu')(dence6)
dence8 = Dense(300, activation= 'linear')(dence7)
dence9 = Dense(200, activation= 'linear')(dence8)
dence10 = Dense(100, activation= 'linear')(dence9)
output1 = Dense(1, activation= 'linear')(dence10)
model = Model(inputs = input1, outputs = output1)
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse' , optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor= 'val_loss', 
                             patience= 100, mode = 'min', 
                             restore_best_weights=True, 
                             verbose=1)
import datetime
date = datetime.datetime.now()
print(date) #2023-01-12 14:57:51.908060
print(type(date)) #class 'datetime.datetime' 
date = date.strftime("%m%d_%H%M") #0112_1502 ->스트링 문자열 형식으로 바꿔주기
print(date)
print(type(date)) #<class 'str'>->스트링 문자열 형태임 

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # epoch는 정수 4자리까지 val_loss는 소수점 4자리 이하까지 .hdf5 파일 만들기

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, #가장 좋은 지점을 저장
                      filepath= filepath +'k30'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, 
        callbacks=[es, mcp], verbose=1)

# model.save(path + 'keras30_ModelCheckPoint13_save_model.h5')

# model = load_model(path +'MCP/keras30_ModelCheckPoint1.hdf5')

#4.평가, 예측
print("=====================1. 기본 출력==============================")
mae, mse = model.evaluate(x_test, y_test)
print('mse : ', mse)
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("***********************************")
print("R2 스코어 : ", r2)
print("***********************************")


