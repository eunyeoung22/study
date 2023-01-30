import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]).transpose()

print(x1_datasets.shape) #(100, 2) 삼성전자 시가, 고가

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T
print(x2_datasets.shape) #(100, 3) 아모레 시가, 고가, 종가
x3_datasets = np.array([range(101,201), range(1301,1401)]).T
print(x3_datasets.shape) #(100, 2)

y = np.array(range(2001,2101))
print(y.shape)#(100,)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
   x1_datasets, x2_datasets, x3_datasets, y, train_size=0.7, random_state=123
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape)#(70, 2) (70, 3) (70, 2) (70,)


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
dence22 = Dense(100, activation='relu', name = 'ds22')(dence21)
output2 = Dense(100, activation='relu', name = 'ds23')(dence22)

#2-3 모델3.
input3 = Input(shape=(2,))
dence31 = Dense(100, activation='relu', name = 'ds31')(input3)
dence32 = Dense(100, activation='relu', name = 'ds32')(dence31)
output3 = Dense(100, activation='relu', name = 'ds33')(dence32)

#2-3 모델병합
from tensorflow.keras.layers import concatenate,Concatenate
merge1 = Concatenate([output1,output2,output3], name ='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1,input2,input3], outputs=last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=10)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print(loss)

