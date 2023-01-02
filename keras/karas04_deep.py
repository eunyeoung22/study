import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. (정제된) 데이터 
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])


#2. 모델 구성 
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(50))
model.add(Dense(4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=32)

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)

