import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
# print(range(10))
x = x.T
print(x.shape)   #(3, 10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = y.T
print(y.shape)

model=Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))

model.compile(loss='mae' , optimizer='adam')
model.fit(x,y, epochs=300, batch_size=2)

loss = model.evaluate(x,y)
print('loss : ' , loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 결과 : ' , result)



"""
결과 값 - [9,30,210]의 결과 :  [[5.97873   1.3720586]]
최초 loss : 1.4379

마지막 결과값
loss :  0.14602626860141754
[9,30,210]의 결과 :  [[10.028992   1.4373764]]
"""