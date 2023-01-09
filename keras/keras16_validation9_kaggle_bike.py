import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#1. 데이터
path = './_data/bike/' #  데이터 위치
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

# print(train_csv.shape)  #(10886, 11)
# print(test_csv.shape)  #(6493, 8)
# print(submission.shape) #(6493, 1)
# print(train_csv.info())
# print(test_csv.info())

x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)
print(y.shape) #(10,886,0) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9 , shuffle=True, random_state=123
)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation = 'relu'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(90, activation = 'linear'))
model.add(Dense(80, activation = 'linear'))
model.add(Dense(50, activation = 'linear'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(10, activation = 'relu')) 
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
import time
model.compile(loss='mse' , optimizer='adam')
<<<<<<< HEAD
start = time.time()
model.fit(x_train, y_train, epochs=1500, batch_size=10, validation_split=0.25)
end = time.time()
=======

model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.25)

>>>>>>> 2183001bb1b40d3cf201de809144c8e75c5ffa90

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predict : ' , y_predict)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))**0.5

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("***********************************")

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission['count'] = y_submit
print(submission)
<<<<<<< HEAD
submission.to_csv(path + 'submission_01090931.csv')
=======
submission.to_csv(path + 'submission_01082152.csv')
>>>>>>> 2183001bb1b40d3cf201de809144c8e75c5ffa90


"""
RMSE :  168.51801356609832 #relu
RMSE :  149.02541250872042
RMSE :  149.10960514117627 #last
<<<<<<< HEAD
RMSE :  150.33808468361812
=======
RMSE : 12.195875452342916
RMSE :  16.15427168435242
RMSE :  12.336959716845895
RMSE :  12.404882025716235
>>>>>>> 2183001bb1b40d3cf201de809144c8e75c5ffa90
"""