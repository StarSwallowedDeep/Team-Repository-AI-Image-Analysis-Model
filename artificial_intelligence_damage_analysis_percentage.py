import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

import pymongo
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# 함수로 이미지 가져오기
def get_image_from_mongodb():
    # MongoDB에 연결
    client = pymongo.MongoClient("mongodb://localhost:27017")
 
    # 데이터베이스 선택
    db = client["market"]

    # 컬렉션 선택
    collection = db["posts"]

    results = collection.find({}, {"photo": 1}).sort([("_id", pymongo.DESCENDING)]).limit(1)

    for result in results:
        if "photo" in result:
            # ID를 제외하고 실제 이미지 데이터를 가져옴
            image_data = result["photo"]

            # Base64 디코딩
            image_data = image_data.split(",")[-1]
            image_data = base64.b64decode(image_data)

            # BytesIO로 읽어옴
            image_bytes = BytesIO(image_data)

            # BytesIO에서 이미지 열기
            img_io = Image.open(image_bytes)
            img_io.show()
            
            global image_id
            image_id = result["_id"]
            
            return img_io, collection

# 데이터 경로 및 클래스 수 설정
train_data_dir = 'train'
validation_data_dir = 'validation'
num_classes = 2  # intact와 damaged 두 개의 클래스로 설정

# VGG16 모델 로드 (weights='imagenet'은 사전 학습된 가중치 사용)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# 마지막 fully connected 레이어 변경
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 전체 모델 정의
classification_model = Model(inputs=base_model.input, outputs=predictions)

# 기존 모델 층 동결
for layer in base_model.layers:
    layer.trainable = False

# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 데이터 로더 생성
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['intact', 'damaged']  # 클래스 이름 설정
)

# 모델 컴파일
classification_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 전이 학습 실행
classification_model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))

# U-Net 스타일의 세그멘테이션 모델 정의
inputs = Input(shape=(224, 224, 3))
conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

up3 = concatenate([UpSampling2D(size=(2, 2))(conv2), conv1], axis=-1)
conv3 = Conv2D(64, 3, activation='relu', padding='same')(up3)
conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)

outputs = Conv2D(1, 1, activation='sigmoid')(conv3)  # Binary segmentation

segmentation_model = Model(inputs=inputs, outputs=outputs)

# 세그멘테이션 모델 컴파일
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 이미지 로드 및 전처리
img_io, collection = get_image_from_mongodb()

# 이미지 데이터를 BytesIO로 직접 변환
img_bytes = BytesIO()
img_io.save(img_bytes, format='PNG')
img_bytes.seek(0)  # BytesIO의 포인터를 처음으로 돌려놓음

# BytesIO에서 이미지 열기
img = Image.open(img_bytes)
img = img.convert("RGB")  
img = img.resize((224, 224))

# image_path = 'test_model_D.png'
# img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 이미지 분류 예측
classification_predictions = classification_model.predict(img_array)
predicted_class = np.argmax(classification_predictions, axis=1)[0]

# damaged로 분류된 경우 세그멘테이션 모델을 사용하여 손상된 부분을 예측
if predicted_class == 1:
    segmentation_predictions = segmentation_model.predict(img_array)
    
    # 시각화를 위해 0과 1로 이루어진 이진 이미지로 변환
    binary_segmentation = (segmentation_predictions > 0.5).astype(np.uint8)
    
    # damaged 부분을 빨간색으로 표시하는 이미지 생성
    damaged_visualization = np.copy(img_array[0])
    damaged_visualization[binary_segmentation[0, :, :, 0] == 1, 0] = 255  # Red channel
    
    # 파손된 부분을 나타내는 이진 마스크
    damaged_mask = binary_segmentation[0, :, :, 0] 
    damaged_pixel_count = np.sum(damaged_mask)
    
    # 이미지 읽기
    image = np.array(img_io)
    
    # 이미지 처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 제품 영역의 크기 
    total_product_pixel = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        total_product_pixel += w * h
        
    # 손상된 픽셀의 백분율 계산
    damage_percentage = (damaged_pixel_count / total_product_pixel) * 100
    
    print("Damage Percentages:", damage_percentage, "%")
    
    # 시각화
    plt.imshow(damaged_visualization.astype(np.uint8))
    plt.axis('off')
    plt.show()

# 분류 결과 출력
class_names = ['intact', 'damaged']  # 클래스 이름 리스트
predicted_class_name = class_names[predicted_class]
print("Predicted Class:", predicted_class_name)
