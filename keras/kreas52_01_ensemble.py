import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]).transpose()

print(x1_datasets.shape) #(100, 2) 삼성전자 시가, 고가

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T
print(x2_datasets.shape) #(100, 3) 아모레 시가, 고가, 종가

y = np.array(range(2001,2101))
print(y.shape)#(100,)

from sklearn.model_selection import train_test_split
x1_train, x1_tset, x2_train, x2_test, y_train, y_test = train_test_split(
   x1_datasets, x2_datasets, y, train_size=0.7, random_state=123
)

print(x1_train.shape, x2_train.shape, y_train.shape)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1.
input1 = Input(shape=(2,))
dence1 = Dense(100, activation='relu', name = 'ds11')(input1)
dence2 = Dense(100, activation='relu', name = 'ds12')(dence1)
dence3 = Dense(100, activation='relu', name = 'ds13')(dence2)
dence4 = Dense(100, activation='relu', name = 'ds14')(dence3)
output1 = Dense(80, activation='relu', name = 'ds15')(dence4)

#2-2 모델2.
input2 = Input(shape=(3,))
dence21 = Dense(100, activation='relu', name = 'ds21')(input2)
dence22 = Dense(100, activation='relu', name = 'ds22')(dence1)
output2 = Dense(100, activation='relu', name = 'ds23')(dence2)

#2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1,output2], name ='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1,input2], outputs = last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size = 8)

#4. 평가, 예측
loss = model.evaluate([x1_tset, x2_test], y_test)
print(loss)