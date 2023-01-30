import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]).transpose()

print(x1_datasets.shape) #(100, 2) 삼성전자 시가, 고가

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T
print(x2_datasets.shape) #(100, 3) 아모레 시가, 고가, 종가
x3_datasets = np.array([range(100,200), range(1301,1401)]).T
print(x3_datasets.shape) #(100, 2)

y1 = np.array(range(2001,2101))
y2 = np.array(range(201,301))
print(y1.shape)#(100,)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
   x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.7, random_state=123
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape)#(70, 2) (70, 3) (70, 2) (70,) (70,)
print(x1_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape)#(30, 2) (30, 3) (30, 2) (30,) (30,)


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

#2-4 모델병합
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate()([output1,output2,output3], name ='mg1')#Concatenate 사용 시 Concatenate() 
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#분기 시 input_shape는 병합의 마지막 last_output으로 사용한다.
#2-5 모델5. 분기1
dence41 = Dense(100, activation='relu', name = 'ds41')(last_output)
dence42 = Dense(100, activation='relu', name = 'ds42')(dence41)
output4 = Dense(33, activation='relu', name = 'ds43')(dence42)

#2-6 모델5. 분기2
dence51 = Dense(100, activation='relu', name = 'ds51')(last_output)
dence52 = Dense(100, activation='relu', name = 'ds52')(dence51)
output5 = Dense(10, activation='relu', name = 'ds53')(dence52)

model = Model(inputs = [input1,input2,input3], outputs = [output4, output5])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=100, batch_size=10)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print(loss)

"""
[1938172.5, 1906734.375, 31438.150390625]
  로스 합  = y1 로스 + y2 로스

[1677764.125, 1652505.125, 25259.0234375, 808.346923828125, 111.9678726196289]
    로스 합  =  y1 로스 +    y2 로스,         매트릭스1(mae) ,    매트릭스2(mae)
"""