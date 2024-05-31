
### 학습 모델 저장하기 ###

import tensorflow as tf
import numpy as np
import os

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



#쇼핑몰 이미지 데이터셋 가져오기
#텐서플로우라이브러리에서 구글이 호스팅해주는 데이터 셋 중하나 
( (trainX, trainY), (testX, testY) ) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']


# 1. 모델만들기
model = tf.keras.Sequential([
    
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
# 모델 아웃라인 출력
model.summary()


# epochs 돌리는 중간중간 저장하기
call_back= tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint/mnist{epoch}.keras',
    # validation Accuracy거 가장 높은 것만 되는 checkpoint만 저장하기
    monitor='val_acc',
    mode='max',
    save_freq='epoch'
)

# 2. compile 하기
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 3. model.fit 하기 (x와 y데이터를 집어넣어서)
model.fit(trainX,trainY, validation_data=(testX, testY), epochs=3, callbacks=[call_back])

# # 폴더가 존재하지 않으면 생성
# save_dir = 'newFolder'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# 모델 저장
# model.save(os.path.join('newFolder','model1.keras'))

# # # 저장한 모델 불러오기
# load_model = tf.keras.models.load_model('newFolder/model1.keras')
# load_model.summary()

# # 저장모델 평가
# load_model.evaluate(testX, testY)



# 1. 모델만들기
model2 = tf.keras.Sequential([
    
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
# 모델 아웃라인 출력
model2.summary()

# 2. compile 하기
model2.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# w값만 저장해놨으면 모델을 만들고 w값(checkpoint파일)을 로드하면 됨.
model2.load_weights('checkpoint/mnist.keras')

model2.evaluate(testX,testY)