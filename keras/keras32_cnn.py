from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten #Conv2D(이미지(그림))

model = Sequential()
                              # 인풋은(60000 , 5,5,1)
model.add(Conv2D(filters=10, kernel_size=(2,2),#kernel_size : 조각내는 기준
          input_shape=(5, 5, 1)))# filter : 히든레이어의 갯수 개념 (N,4,4,10) = 행은 무시해도 되기에 60000개라도 'N'으로 표시해도됨
          #(batch_size, rows, colums, channels)
model.add(Conv2D(filters=5, kernel_size=(2,2))) #(3,3,5) = (N,3,3,5)
#model.add(Conv2D(5,(2,2)))
model.add(Flatten())#dence 형식으로 바꾸기(행렬) #(45,) = (N,45)
model.add(Dense(units=10))
model.add(Dense(units=10))
model.add(Dense(units=10))
model.add(Dense(units=10))
         # 인풋은 (batch_size, input_dim)
model.add(Dense(4, activation='relu'))
model.summary() #파라미터 


                    