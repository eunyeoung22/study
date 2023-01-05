import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/' #데이터 위치
train_csv = pd.read_csv(path + 'train.csv' , index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv)
# print(train_csv.shape) #(1459, 9) -> 열 10이지만 'count'는 y값이므로 1을 빼준다.
# print(train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',        
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],   
# #       dtype='object')
# print(train_csv.info())
# print(test_csv.info())
# print(train_csv.describe())
print(submission.shape) #(715, 1)


## 결측치 처리 1. 제거 ##
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape) #(1328, 10)
x = train_csv.drop(['count'], axis=1)
print(x) #[1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape) #(1328,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9, shuffle=True, random_state=123
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델 구성

model = Sequential()
model.add(Dense(1000, input_dim = 9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse' , optimizer='adam', metrics='mae')
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)


def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))

print("걸린시간 : "  , end - start)
print("***********************************")

"""
1. CPU
RMSE :  98.73692443472156
걸린시간 :  156.6539490222931

2. GPU
RMSE :  76.25947914054537
걸린시간 :  150.39956283569336
"""

"""
RMSE :  48.99774291529566
RMSE :  48.538199343440056
RMSE :  52.696013567460675
RMSE :  52.86050457047641
"""

# .to_csv()를 사용하여 
# submission_0105.csv를 완성하시오.
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(715, 1)

submission['count'] = y_submit
print(submission)

# submission.to_csv(path + 'submission_01050326.csv')





