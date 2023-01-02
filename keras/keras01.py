import tensorflow as tf
#텐서플로를 임포트 합니다. 하지만 너무 길어서 as 뒤는 줄여준다.

print(tf.__version__)

#(주석)

import numpy as np


#01.데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])


#02.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(1,input_dim=1))

#03.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=2500)

#04.평가, 예측
result = model.predict(6)
print('결과:', result)
