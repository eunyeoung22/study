import numpy as np
x_datasets = np.array([range(100), range(301, 401)]).transpose()

print(x_datasets.shape) #(100, 2) 삼성전자 시가, 고가


y1 = np.array(range(2001,2101))
y2 = np.array(range(201,301))
print(y1.shape)#(100,)


from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
   x_datasets, y1, y2, train_size=0.7, random_state=123
)

print(x_train.shape, y1_train.shape, y2_train.shape)#70, 2) (70,) (70,)
print(x_test.shape, y1_test.shape, y2_test.shape)#(30, 2) (30,) (30,)



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


model = Model(inputs = [input1], outputs = [output1])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x_train], [y1_train, y2_train], epochs=100, batch_size=10)

#4. 평가, 예측
loss = model.evaluate([x_test], [y1_test, y2_test])
print(loss)

