import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

a = np.array(range(1,101))
x_predict = np.array(range(96,106))


timestemps = 5 #x는 4개, y는 1개

def split_x(dataset, timestemps) : 
    aaa = []
    for i in range(len(dataset) - timestemps + 1) :
        subset = dataset[i : (i+ timestemps)]
        aaa.append(subset)
    return np.array(aaa) # numpy type으로 리턴해주기 위함

bbb = split_x(a, timestemps) 
print(bbb)
print(bbb.shape)#(96, 5)

x = bbb[:, :-1]# 전체에서 끝자리만 빼고 전체 
y = bbb[:, -1]# 전체에서 끝자리만
print(x,y)

print(x.shape, y.shape) #(96, 4) (96,)
x = x.reshape(96,4,1)
print(x.shape, y.shape)

#모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(4,1), activation='relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
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

x_predict= split_x(x_predict, 4)
print(x_predict.shape)

x_predict = x_predict.reshape(7,4,1)

result = model.predict(x_predict)
print('[96,106]의 결과 : ', result)




"""
[[100.00294 ]
 [101.00554 ]
 [102.00807 ]
 [103.01059 ]
 [104.013084]
 [105.01556 ]
 [106.018   ]]
"""


