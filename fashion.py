import tensorflow as tf
import matplotlib.pyplot as plt


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

print(trainX[0])
#.shape: 데이터 갯수
print(trainX.shape)

# 1. 모델만들기
# 2. compile 하기
# 3. model.fit 하기 (x와 y데이터를 집어넣어서)

print(trainY)

plt.imshow(trainX[0])
plt.gray()
plt.colorbar()
plt.show()
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']
