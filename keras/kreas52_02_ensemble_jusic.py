import numpy as np
import pandas as pd


#1.데이터
path = './_data/jusic/' #데이터 위치가 현재 작성파일과 동일한 위치에 있을때(현재위치)
x1_datasets = pd.read_csv(path + 'samsung.csv', encoding = 'euc-kr')
x2_datasets = pd.read_csv(path + 'amore.csv', encoding = 'euc-kr')
print(x1_datasets.shape, x2_datasets.shape)#(1980, 17) (2220, 17)


y = x1_datasets['시가']
print(y)
print(y.shape) #(1980,)


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

x1_train, x1_tset, x2_train, x2_test, y_train, y_test = train_test_split(
   x1_datasets, x2_datasets, y, train_size=0.7, random_state=123
)



# #2. 모델구성

# #2-1 모델1.
input1 = Input(shape=(17,))
dence1 = Dense(100, activation='relu', name = 'ds11')(input1)
dence2 = Dense(100, activation='relu', name = 'ds12')(dence1)
dence3 = Dense(100, activation='relu', name = 'ds13')(dence2)
dence4 = Dense(100, activation='relu', name = 'ds14')(dence3)
output1 = Dense(80, activation='relu', name = 'ds15')(dence4)

# #2-2 모델2.
input2 = Input(shape=(17,))
dence21 = Dense(100, activation='relu', name = 'ds21')(input2)
dence22 = Dense(100, activation='relu', name = 'ds22')(dence1)
output2 = Dense(100, activation='relu', name = 'ds23')(dence2)

# #2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1,output2], name ='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1,input2], outputs = last_output)
model.summary()

# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size = 8)

# #4. 평가, 예측
result = model.evaluate([x1_tset, x2_test], y_test)
print(result)