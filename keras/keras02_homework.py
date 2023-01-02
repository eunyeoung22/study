import tensorflow as tf


import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_dim=1))

model.compile(loss='mae' , optimizer='adam')
model.fit(x, y, epochs=2000)

result = model.predict([6])
print('결과 :' , results)
