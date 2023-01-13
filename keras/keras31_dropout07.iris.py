from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
y = to_categorical(y) # 노드 마지막 아웃풋이 개수 오류에 대한 치환을 해줄 경우
# print(y)
# print(y.shape)
# print(datasets.DESCR) #판다스 .describe() / .info() 사용
# print(datasets.feature_names) #판다스 .columns 사용
# print(x.shape, y.shape) #(150, 4) (150,)
x_train, x_test, y_train, y_test = train_test_split(
                                x, y, shuffle = True, #False일 경우 문제점 : y_test와 y_train 성능에 문제발생
                                random_state=123, 
                                test_size=0.2,
                                stratify=y) # y에 대한 경우에만 사용(균일하게 분배)

# print(y_train)
# print(y_test)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

# 2. 모델 구성(순차형)
# model = Sequential()
# model.add(Dense(50, input_shape = (4,), activation = 'relu'))
# model.add(Dense(40, activation = 'linear'))
# model.add(Dense(30, activation = 'linear'))
# model.add(Dense(20, activation = 'relu'))
# model.add(Dense(30, activation = 'linear'))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(30, activation = 'linear'))
# model.add(Dense(30, activation = 'linear'))
# model.add(Dense(30, activation = 'sigmoid'))
# model.add(Dense(3, activation = 'softmax'))#다중분류 시 activation = 'softmax' 사용, 노드개수 3 : y 클래스 종류
#                                            #최종 아웃풋 레이어에 액티베이션은 softmax다
# model.summary()
# # Total params: 8,583
# # Trainable params: 8,583
# # Non-trainable params: 0

# 2. 모델 구성(함수형)
input1 = Input(shape=(4,))
dence1 = Dense(50, activation= 'relu')(input1)
dence2 = Dense(40, activation= 'sigmoid')(dence1)
dence3 = Dense(30, activation= 'linear')(dence2)
dence4 = Dense(20, activation= 'relu')(dence3)
drop4 = Dropout(0.3)(dence4)
dence5 = Dense(30, activation= 'linear')(drop4)
dence6 = Dense(30, activation= 'relu')(dence5)
dence7 = Dense(30, activation= 'linear')(dence6)
dence8 = Dense(30, activation= 'linear')(dence7)
drop8 = Dropout(0.3)(dence8)
dence9 = Dense(30, activation= 'sigmoid')(drop8)
output1 = Dense(3, activation= 'softmax')(dence9)
model = Model(inputs = input1, outputs = output1)
model.summary()
# Total params: 8,583
# Trainable params: 8,583
# Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

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
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2,
                callbacks=[es, mcp], verbose=1)
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)


"""
loss:  0.22739247977733612
accuracy :  0.8666666746139526
[1 2 0 0 2 2 0 2 0 2 2 2 0 1 0 1 2 2 2 1 2 0 0 1 0 0 1 1 2 1]
[1 2 0 0 2 2 0 1 0 2 2 1 0 1 0 1 2 2 2 2 2 0 0 1 0 0 1 1 1 1]
0.8666666666666667

drop
loss:  0.3513701260089874
accuracy :  0.8999999761581421
"""