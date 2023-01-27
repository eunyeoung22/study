import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM

#1.데이터
dataset = np.array({1,2,3,4,5,6,7,8,9,10}) # (10,)

#y = ?? y값은 주어지지 않고 새로 x값으로 나누어 작업해준다.(시계열 데이터:순차열)
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)#(7, 3) (7,) #-> [[[1],[2],[3]],
                                    #    [[2],[3],[4]], ...]

x = x.reshape(7,3,1)#7개의 나열 값에서 3개 묶음을 1개씩 훈련시킬것이다.
print(x.shape)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape =(3,1), activation= 'relu')) #순차적으로 연산하기 때문에 Flatten안해줘도 된다.
                                            #(N , 3, 1) -> ([batch, timestemps, feature]
model.add(LSTM(units=10, input_shape =(3,1))) 
model.add(Dense(5, activation= 'relu'))
model.add(Dense(1))

#심플
#10 * (10 + 1 + 1) = 120
#units *(units + features + bias) = param

#LSTM
#10 * (10 + 1 + 1) = 480


model.summary()
