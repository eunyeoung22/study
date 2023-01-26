import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

#1.데이터

a = np.array(range(1,14))
timestemps = 3

def split_x(dataset, timestemps) :
    aaa =[]
    for i in range(len(dataset) - timestemps + 1) :
        subset = dataset[i : (i + timestemps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timestemps)
print(bbb)

x = bbb[:,:-1]
y = bbb[:,-1]
print(x,y)

print(x.shape, y.shape)#(11, 2) (11,)
x = x.reshape(11,2,1)
print(x.shape, y.shape)

# 2.모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(2,1), return_sequences= True, activation='relu'))
model.add(LSTM(100, return_sequences= True, activation='relu'))
model.add(LSTM(100))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(30, activation= 'linear'))
model.add(Dense(20, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))

# 3.컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs = 500)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print('loss 결과: ', loss)
y_predict = np.array([12, 13]).reshape(1,2,1)
result = model.predict(y_predict)
print('[12, 13] 결과 : ', result)

"""
loss 결과:  0.0006607401301153004
[12, 13] 결과 :  [[13.907418]]

loss 결과:  0.000123544130474329
[12, 13] 결과 :  [[13.982805]]
"""
