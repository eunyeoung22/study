
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape)#(9797, 8) (1089, 8)
x_train = x_train.reshape(9797,4,2)
x_test = x_test.reshape(1089,4,2)

# # print(x)
# # print(type(x)) #<class 'numpy.ndarray'>
# # print('최소값 : ', np.min(x))
# # print('최대값 : ',np.max(x))
              
#2. 모델구성(순차형)
model = Sequential()
model.add(Conv1D(100, 2, input_shape =(4,2), activation ='relu'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(100, activation = 'linear'))
model.add(Dense(90, activation = 'linear'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(50, activation = 'linear'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
# model.summary()
# # Total params: 54,081
# # Trainable params: 54,081
# # Non-trainable params: 0



#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss',
                              mode = 'min',
                              patience = 350,
                              restore_best_weights=True,
                              verbose=1)
# import datetime
# date = datetime.datetime.now()
# print(date) #2023-01-12 14:57:51.908060
# print(type(date)) #class 'datetime.datetime' 
# date = date.strftime("%m%d_%H%M") #0112_1502 ->스트링 문자열 형식으로 바꿔주기
# print(date)
# print(type(date)) #<class 'str'>->스트링 문자열 형태임 

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # epoch는 정수 4자리까지 val_loss는 소수점 4자리 이하까지 .hdf5 파일 만들기

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True, #가장 좋은 지점을 저장
#                       filepath= filepath +'k31_05_'+ '_' + date + '_' + filename) #파일 저장 경로 지정                
#                     #   filepath= path +'MCP/keras30_ModelCheckPoint13.hdf5') #파일 저장 경로 지정
model.fit(x_train, y_train, epochs=3000, batch_size=32, validation_split=0.2,
                callbacks=[es], verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predict : ' , y_predict)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

    

print("***********************************")
print("RMSE : ", RMSE(y_test, y_predict))
print("***********************************")

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6)) #그래프 사이즈
# plt.plot(hist.history['loss'], c = 'red', marker = '.', 
#         label='loss') # c : 그래프 선 color / label : 그래프 선 이름
# plt.plot(hist.history['val_loss'], c = 'blue', marker = '.', 
#         label = 'val loss')
# plt.grid() #격자
# plt.xlabel('epochs') #x축
# plt.ylabel('loss') #y축
# plt.title('kaggle loss') # 그래프 타이틀
# plt.legend() # 범례(알아서 빈곳에 현출)
# # plt.legend(loc='upper right') #범례(그래프 오른쪽)
# # plt.legend(loc='upper left') #범례(그래프 왼쪽)
# plt.show() # 그래프 보여줘

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission['count'] = y_submit
print(submission)
submission.to_csv(path + 'submission_01130213.csv')


"""   
RMSE :  154.47338779305323
RMSE :  152.27039138022755

적용후
RMSE :  153.80548813233713
"""