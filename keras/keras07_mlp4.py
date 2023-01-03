import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array(range(10)) #(10,) (10,1)

print(x.shape)  #(10,)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = y.T
print(y.shape)

model=Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss='mae' , optimizer='adam')
model.fit(x,y, epochs=300, batch_size=1)

loss = model.evaluate(x,y)
print('loss : ' , loss)

result = model.predict([9])
print('[9]의 결과 : ' , result)



"""
loss :  0.11339142173528671
[9]의 결과 :  [[10.018262   1.7659357  0.2887626]]
"""