import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

a = np.array(range(1,101))
x_predict = np.array(range(96,106))

timestemps1 = 5 #x는 4개, y는 1개
timestemps2 = 4
def split_x(dataset, timestemps) : 
    aaa = []
    for i in range(len(dataset) - timestemps + 1) :
        subset = dataset[i : (i+ timestemps)]
        aaa.append(subset)
    return np.array(aaa) # numpy type으로 리턴해주기 위함

bbb = split_x(a, timestemps1) 
print(bbb)
print(bbb.shape)#(96, 5)

x_predict = split_x(x_predict, timestemps2)
print(x_predict)
print(x_predict.shape)#(7, 4)

x = bbb[:, :-1]# 전체에서 끝자리만 빼고 전체 
y = bbb[:, -1]# 전체에서 끝자리만
print(x,y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123
)
print(x_train.shape, x_test.shape)#(72, 4) (24, 4)

#모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(4,)))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(80, activation= 'linear'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(30, activation= 'linear'))
model.add(Dense(20, activation= 'linear'))
model.add(Dense(1, activation= 'linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 500)

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print(loss)
result = model.predict(x_predict)
print('예측 결과 : ', result)




"""
[[100.00003 ]
 [101.00003 ]
 [102.00006 ]
 [103.000046]
 [104.00004 ]
 [105.000046]
 [106.00003 ]]
"""


