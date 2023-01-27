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

# feature 수를 늘리는 경우 reshape 부분만 수정해서 실행 가능 
# 초기 feature 값과 동일하게 맞춰서 수정해야 한다.
x_train = x_train.reshape(72, 2, 2)
x_test = x_test.reshape(24, 2, 2)
x_predict = x_predict.reshape(7,2,2)

#모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(2,2), activation='relu'))
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
model.fit(x_train, y_train, epochs = 500)

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print(loss)
result = model.predict(x_predict)
print('예측 결과 : ', result)

"""
[[ 99.96285 ]
 [100.95504 ]
 [101.946625]
 [102.9378  ]
 [103.928604]
 [104.91903 ]
 [105.909096]]
 """







