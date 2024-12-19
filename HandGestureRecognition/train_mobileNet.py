import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_model_optimization as tfmot

ANN_DIR = "ann"
DATASET_DIR = "dataset"
GESTURES = ["one", "peace", "stop", "ok", "rock"]
INTPUT_SHAPE = 96


# 加载 JSON 注释
def load_annotations():
    annotations = []
    for gesture in GESTURES:
        ann_path = os.path.join(ANN_DIR, f"{gesture}.json")
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
            for image_id, data in ann_data.items():
                file_path = os.path.join(DATASET_DIR, f"train_val_{gesture}", f"{image_id}.jpg")
                if os.path.exists(file_path):
                    for bbox, label in zip(data["bboxes"], data["labels"]):
                        if label != "no_gesture":
                            annotations.append((file_path, label, bbox))
    return annotations


# 图像裁剪函数
def crop_image(img, bbox):
    width, height = img.size
    x, y, w, h = bbox
    left = int(x * width)
    top = int(y * height)
    right = left + int(w * width)
    bottom = top + int(h * height)
    return img.crop((left, top, right, bottom))


# 数据加载和增强
def load_data(annotations, img_size=(INTPUT_SHAPE, INTPUT_SHAPE)):
    images, labels = [], []
    label_map = {gesture: idx for idx, gesture in enumerate(GESTURES)}  # 手势映射到索引

    for file_path, label, bbox in annotations:
        if label not in label_map:
            continue
        img = Image.open(file_path).convert('L')  # 转换为灰度图
        img = crop_image(img, bbox)
        img = img.resize(img_size)
        img = np.expand_dims(np.array(img) / 255.0, axis=-1)  # 增加通道维度
        images.append(img)
        labels.append(label_map[label])

    return np.array(images), np.array(labels)


# 加载注释并处理数据
annotations = load_annotations()
images, labels = load_data(annotations)
train_size = int(0.8 * len(images))
train_images, val_images = images[:train_size], images[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)
train_dataset = datagen.flow(train_images, train_labels, batch_size=32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)


# 构建 MobileNetV2 模型
def build_mobilenet_model(input_shape=(INTPUT_SHAPE, INTPUT_SHAPE, 1), num_classes=len(GESTURES), alpha=0.35):
    base_model = MobileNetV2(
        weights=None,  # 不使用预训练权重，重新训练模型
        include_top=False,
        input_shape=input_shape,
        alpha=alpha
    )
    base_model.trainable = True  # 允许训练所有层

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()
    return model


# 构建与编译模型
base_model = build_mobilenet_model()

base_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# 回调函数：包括学习率调度
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# 模型训练
history = base_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks,
    steps_per_epoch=len(train_images) // 32,
    validation_steps=len(val_images) // 32
)


# 保存模型为 tflite
def save_model_as_tflite(model, filename="gesture_model_mobilenetv2_gray.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model)


# 定义校准数据集
def representative_dataset():
    for i in range(100):
        yield [np.expand_dims(train_images[i].astype(np.float32), axis=0)]


# 保存模型为 INT8 量化的 TFLite
def save_model_as_int8_tflite(model, filename="gesture_model_mobilenetv2_gray_int8.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model)


# 调用保存 INT8 量化模型的函数
save_model_as_int8_tflite(base_model)
print("优化后的灰度图 INT8 量化模型已保存为 gesture_model_mobilenetv2_gray_int8.tflite")
