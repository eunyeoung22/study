import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#1.데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
y = to_categorical(y)

# print(x.shape, y.shape) #(1797, 64) (1797,)
# print(np.unique(y, return_counts=True))
# #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# #array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(
                x, y, shuffle=True,
                random_state=123,
                test_size=0.2,
                stratify=y
)
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)#(1437, 64) (360, 64)
x_train = x_train.reshape(1437,8,8)
x_test = x_test.reshape(360,8,8)



# print(x)
# print(type(x)) #<class 'numpy.ndarray'>
# print('최소값 : ', np.min(x))
# print('최대값 : ',np.max(x))

#2. 모델 구성(순차형)
model = Sequential()
model.add(Conv1D(100, 2,input_shape=(8,8), activation= 'relu'))
model.add(LSTM(100, activation = 'relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='softmax'))
model.summary()
# Total params: 57,810
# Trainable params: 57,810
# Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor= 'val_accurucy',
                              mode = 'max',
                              patience = 70,
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
                      filepath= filepath +'k31_09_'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
                    #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=100, batch_size=20,
                 callbacks=[es, mcp], validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)

"""
loss:  0.8500062823295593
accuracy :  0.9472222328186035

cnn 적용 후
loss:  0.3630369007587433
accuracy :  0.9777777791023254

LSTM 적용 후
loss:  0.22798903286457062
accuracy :  0.9694444537162781
"""