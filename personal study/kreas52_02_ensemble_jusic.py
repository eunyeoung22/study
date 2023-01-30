import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1.데이터
path = './_data/jusic/' #데이터 위치가 현재 작성파일과 동일한 위치에 있을때(현재위치)
sam = pd.read_csv(path + 'samsung.csv', encoding = 'euc-kr', header=0, index_col=0, 
                 nrows= 1166, usecols=[0,1,2,3,4,8])
amor = pd.read_csv(path + 'amore.csv', encoding = 'euc-kr', index_col=0,
                  nrows= 1166, usecols=[0,1,2,3,4,8], header=0)
# ecoding : csv 파일 내 한글이 포함되어 있어 인코딩 필요(eocoding = 'cp949' 또는 'euc-kr')
# header : 컬럼의 이름을 확인하기 위해 상단 "0" 헤더 지정을 위해
# index_col : 해당 컬럼을 인덱스로 지정하여 컬럼개수에 포함되지 않음
# nrows : csv 파일 내 필요한 정보(행)만 가져오기 위해
# usecols : 필요한 열 정보만 가져오기 위해
print(sam)
print(sam.shape, amor.shape)#(1167, 6) (1903, 6)
print(sam.info())

# string 데이터 타입을 수치형(int, double, float 등) 데이터 타입으로 수정
# 1. 삼성
sam['시가'] = sam['시가'].str.replace(',', '').astype('float')
sam['고가'] = sam['고가'].str.replace(',', '').astype('float')
sam['저가'] = sam['저가'].str.replace(',', '').astype('float')
sam['종가'] = sam['종가'].str.replace(',', '').astype('float')
sam['거래량'] = sam['거래량'].str.replace(',', '').astype('float')
# sam['금액(백만)'] = sam['금액(백만)'].str.replace(',', '').astype('float')


# 2. 아모레
amor['시가'] = amor['시가'].str.replace(',', '').astype('float')
amor['고가'] = amor['고가'].str.replace(',', '').astype('float')
amor['저가'] = amor['저가'].str.replace(',', '').astype('float')
amor['종가'] = amor['종가'].str.replace(',', '').astype('float')
amor['거래량'] = amor['거래량'].str.replace(',', '').astype('float')
# amor['금액(백만)'] = amor['금액(백만)'].str.replace(',', '').astype('float')

# sort 일자가 현재 기준으로 정렬되어 있기에 순차형 시 과거 데이터로 될 수 있으므로 오름차순으로 바꾸기
sam = sam.sort_values(['일자'], ascending=[True])
amor = amor.sort_values(['일자'], ascending=[True])

print(sam)
print(amor)

sam_open = sam['시가'][1:]

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Flatten

x1_train, x1_test, y_train, y_test = train_test_split(
   sam[:1165], sam_open, train_size=0.7, shuffle=True, random_state=123
)

x1_train = x1_train.to_numpy()
x1_test = x1_test.to_numpy()

print(x1_train.shape, x1_test.shape)

x1_train = x1_train.reshape(815,5,1)
x1_test = x1_test.reshape(350,5,1)

x2_train, x2_test = train_test_split(
   amor[:1165], train_size=0.7, shuffle=True, random_state=123
)

print(x1_train.shape, x1_test.shape)#(815, 5) (350, 5)
print(x2_train.shape, x2_test.shape)#(815, 5) (350, 5)



# #2. 모델구성

# #2-1 모델1.
input1 = Input(shape=(5,1))
lstm = LSTM(100, input_shape=(5,1), activation = 'relu')(input1)
dence1 = Dense(100, activation='relu', name = 'ds11')(lstm)
flat = Flatten()(dence1)
dence2 = Dense(100, activation='relu', name = 'ds12')(flat)
dence3 = Dense(100, activation='relu', name = 'ds13')(dence2)
dence4 = Dense(100, activation='relu', name = 'ds14')(dence3)
dence5 = Dense(100, activation='relu', name = 'ds15')(dence4)
dence6 = Dense(100, activation='relu', name = 'ds16')(dence5)
output1 = Dense(50, activation='relu', name = 'ds17')(dence6)

# #2-2 모델2.
input2 = Input(shape=(5,))
dence21 = Dense(100, activation='relu', name = 'ds21')(input2)
dence22 = Dense(100, activation='relu', name = 'ds22')(dence21)
dence23 = Dense(100, activation='relu', name = 'ds23')(dence22)
dence24 = Dense(100, activation='relu', name = 'ds24')(dence23)
dence25 = Dense(100, activation='relu', name = 'ds25')(dence24)
dence26 = Dense(100, activation='relu', name = 'ds26')(dence25)
output2 = Dense(50, activation='relu', name = 'ds27')(dence26)

# #2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1,output2], name ='mg1')
merge2 = Dense(150, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
merge4 = Dense(100, name='mg4')(merge3)
merge5 = Dense(100, name='mg5')(merge4)
merge6 = Dense(100, name='mg6')(merge5)
merge7 = Dense(100, name='mg7')(merge6)
merge8 = Dense(100, name='mg8')(merge7)
merge9 = Dense(50, name='mg9')(merge8)


last_output = Dense(1, name='last')(merge5)

model = Model(inputs = [input1,input2], outputs = last_output)
model.summary()

# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=250, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, 
                      filepath='./_save/MCP/sam_open_MCP.hdf5') #파일 저장 경로 지정  
model.fit([x1_train, x2_train], y_train, epochs=500, validation_split=0.2,
           callbacks=[es, mcp], batch_size = 32)

# #4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print(loss)

result = model.predict([sam[1165:].to_numpy().reshape(1,5,1),amor[1165:]]) # train data type 과predict data type 맞추기
print("samsung 시가 : ", result)

"""
samsung 시가 :  [[64489.69]]
"""