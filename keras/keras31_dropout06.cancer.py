import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_breast_cancer()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
# print(x.shape , y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

#2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50,  activation= 'linear', input_shape=(30,)))
# model.add(Dense(40, activation= 'relu'))
# model.add(Dense(30, activation= 'relu'))
# model.add(Dense(10, activation= 'relu'))
# model.add(Dense(1, activation= 'sigmoid')) # 이진분류에선 꼭 마지막에 sigmoid 사용
# model.summary()
# # Total params: 5,141
# # Trainable params: 5,141
# # Non-trainable params: 0

# 2. 모델 구성(함수형)
input1 = Input(shape=(30,))
dence1 = Dense(50, activation= 'linear')(input1)
dence2 = Dense(40, activation= 'relu')(dence1)
drop2 = Dropout(0.5)(dence2)
dence3 = Dense(30, activation= 'relu')(drop2)
dence4 = Dense(10, activation= 'relu')(dence3)
output1 = Dense(1, activation= 'sigmoid')(dence4)
model = Model(inputs = input1, outputs = output1)
model.summary()
# Total params: 5,141
# Trainable params: 5,141
# Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy']) # 이진분류에선 꼭 이 훈련 쓰기
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
es = EarlyStopping(monitor= 'val_loss' ,
                              mode='min',
                              patience = 300, 
                              restore_best_weights=True, # patience 10번 중 마지막 최소값이 아닌 그중 제일 최소값 반환
                              verbose=1
                              )
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
                      filepath= filepath +'k31_05_'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=3000, batch_size=32, validation_split=0.2,
                callbacks=[es, mcp], verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
intarr = list(map(int, y_predict)) 
print(intarr[:10])  #--> 정수형으로 바꿔줘야함 
print(y_test[:10])

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, intarr)
print("accuracy_score : ", acc)


"""
loss :  0.08679293096065521
accuracy :  0.9824561476707458
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1 1 0 1 0 1 1 0 1 1]
accuracy_score :  0.3684210526315789
"""