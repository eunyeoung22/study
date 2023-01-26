import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU

#1.데이터
dataset = np.array({1,2,3,4,5,6,7,8,9,10}) # (10,)

#y = ?? y값은 주어지지 않고 새로 x값으로 나누어 작업해준다.(시계열 데이터:순차열)
x = np.array([[1,2,3],[2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)#(7, 3) (7,) #-> [[[1],[2],[3]],
                                    #    [[2],[3],[4]], ...]

x = x.reshape(13,3,1)#7개의 나열 값에서 3개 묶음을 1개씩 훈련시킬것이다.
print(x.shape)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape =(3,1), activation= 'relu')) #순차적으로 연산하기 때문에 Flatten안해줘도 된다.
                                            #(N , 3, 1) -> ([batch, timestemps, feature]
model.add(GRU(units=10, input_shape =(3,1), activation='relu')) 
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(100, activation= 'linear'))
model.add(Dense(80, activation= 'linear'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(10, activation= 'linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=500)

#4. 평가, 예측
loss = model.evaluate(x,y)
print(loss)
y_pred = np.array([50,60,70]).reshape(1,3,1)

result = model.predict(y_pred)
print('[50,60,70]의 결과 : ', result)

"""
[50,60,70]의 결과 :  [[74.454216]]
[50,60,70]의 결과 :  [[80.376976]]
[50,60,70]의 결과 :  [[79.97844]]
"""