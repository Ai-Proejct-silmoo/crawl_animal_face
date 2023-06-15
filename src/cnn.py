import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dog_data_path = '../asset/dog'
cat_data_path = '../asset/cat'
rabbit_data_path = '../asset/rabbit'

train_ratio = 0.8
test_ratio = 0.2

train_dir = './train'
test_dir = './test'

for data_path, class_name in [(dog_data_path, 'dog'), (cat_data_path, 'cat'), (rabbit_data_path, 'rabbit')]:
    file_list = os.listdir(data_path)
    
    train_files, test_files = train_test_split(file_list, train_size=train_ratio, test_size=test_ratio, random_state=42)

    for file_name in train_files:
        src_path = os.path.join(data_path, file_name)
        dst_path = os.path.join(train_dir, class_name, file_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    
    for file_name in test_files:
        src_path = os.path.join(data_path, file_name)
        dst_path = os.path.join(test_dir, class_name, file_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

classes = ['cat', 'dog', 'rabbit']
image_size = (150, 150)

dataset_path = './train'

def load_dataset():
    data = []
    labels = []
    
    for idx, animal in enumerate(classes):
        folder_path = os.path.join(dataset_path, animal)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            data.append(image)
            labels.append(idx)
    
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    
    return data, labels

data, labels = load_dataset()

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_ratio, random_state=42)
channels = 3

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

loss, accuracy = model.evaluate(test_data, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

print('이미지 url을 입력해주세요.')

test_data_image_path = input();

image = cv2.imread(test_data_image_path)
image = cv2.resize(image, (150, 150))  # 모델에 맞는 이미지 크기로 조정
image = np.expand_dims(image, axis=0)  # 배치 차원 추가
image = image.astype('float32') / 255.0  # 이미지 정규화

predictions = model.predict(image)
class_index = np.argmax(predictions[0])
class_label = '동물_클래스_이름'  # 예측된 동물 클래스에 해당하는 이름

print('예측된 동물 상:', classes[class_index])