import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

a = np.array(range(1,11))
timestemps = 5

def split_x(dataset, timestemps) : 
    aaa = []
    for i in range(len(dataset) - timestemps + 1) :
        subset = dataset[i : (i+ timestemps)]
        aaa.append(subset)
    return np.array(aaa) # numpy type으로 리턴해주기 위함

bbb = split_x(a, timestemps) 
print(bbb)

print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)

print(x.shape, y.shape) #(6, 4) (6,)
x = x.reshape(6,4,1)
print(x.shape, y.shape)

#모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(4,1), return_sequences= True, activation='relu'))
model.add(LSTM(100))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(80, activation= 'linear'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(30, activation= 'linear'))
model.add(Dense(20, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs = 500)

#4.평가, 예측
loss = model.evaluate(x,y)
print(loss)
x_predict = np.array([7,8,9,10]).reshape(1,4,1)

result = model.predict(x_predict)
print('[7,8,9,10]의 결과 : ', result)


#실습
#LSTM 모델구성
"""
[7,8,9,10]의 결과 :  [[10.956064]]
"""