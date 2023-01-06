import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:11]
y_train = y[:11]
x_test = x[11:14] 
y_test = y[11:14] 
x_validation = x[14:17]
y_validation = y[14:17]

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])
# y_validation = np.array([14,15,16])

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation= 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
            validation_data=(x_validation, y_validation))

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측 값 : ", result)



