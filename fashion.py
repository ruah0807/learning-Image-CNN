import tensorflow as tf



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

print(trainX)
