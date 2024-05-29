import os
import shutil
import tensorflow as tf

print(len(os.listdir('train')))

train_dir = 'train'
cat_dir = 'dataset/cat'
dog_dir = 'dataset/dog'
dataset = 'dataset'

# 디렉토리가 존재하지 않으면 생성
# os.makedirs(cat_dir, exist_ok=True)
# os.makedirs(dog_dir, exist_ok=True)

# train 디렉토리 내의 모든 파일을 처리
# for i in os.listdir(train_dir):
#     j = os.path.join(train_dir, i)
#     if 'cat' in i :
#         shutil.copyfile( j, os.path.join(cat_dir, i) )
#     if 'dog' in i :
#         shutil.copyfile( j, os.path.join(dog_dir, i) )
        

train_ds = tf.keras.preprocessing.image_dataset_from_directory (
    dataset, 
    image_size=(64,64),
    # 이미지 2만장 한번에 넣지 않고 batch 숫자만큼 한번에 넣겠소
    batch_size= 64,
    # 20%로 validation 하겠소
    subset='training',
    validation_split = 0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset, 
    image_size=(64,64),
    batch_size= 64,
    subset='validation',
    validation_split = 0.2,
    seed=1234
)
print(train_ds)





def 전처리함수(i, answer) :
    i = tf.cast(i/255, tf.float32)
    return i, answer

train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)


# import matplotlib.pyplot as plt

# for i, answer in train_ds.take(1) :
#     print(i)
#     print(answer)
#     plt.imshow( i[0].numpy().astype('uint8') )
#     plt.show()


# 1. 모델만들기
model = tf.keras.Sequential([
     
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    # 노드 일부를 제거해줌 : overfitting 이 덜 일어나게 됨.
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, input_shape=(64,64), activation='sigmoid'),
])
# 모델 아웃라인 출력
model.summary()

# 2. compile 하기
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# 3. model.fit 하기 (x와 y데이터를 집어넣어서)
model.fit(train_ds, validation_data=val_ds, epochs=5)
