import tensorflow as tf
import numpy as np

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
textX = testX / 255.0

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']


# 1. 모델만들기
model = tf.keras.Sequential([
    
    
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
                    # .MaxPooling2D : 사이즈를 줄이고 중요한 정보를 가운데로 모아주는 함수 
    tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),
                    # Flatten() : 2D or 3D 데이터를 1차원으로 압축해주는 레이어
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, input_shape=(28,28), activation='relu'),
    tf.keras.layers.Dense(10, input_shape=(28,28), activation='softmax'),
])
# 모델 아웃라인 출력
model.summary()

# 2. compile 하기
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 3. model.fit 하기 (x와 y데이터를 집어넣어서)
model.fit(trainX,trainY, validation_data=(textX, testY), epochs=5)

# 4. 모델이 제대로 잘 실행되었는지 평가
# score = model.evaluate(testX, testY)
# print(score)