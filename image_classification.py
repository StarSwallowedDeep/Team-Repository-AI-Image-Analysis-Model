from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# ResNet 모델 로드
model = ResNet50(weights='imagenet')

# 입력 이미지 불러오기
img_path = 'test_img.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# 이미지 분류 예측
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]
print(decoded_predictions)
