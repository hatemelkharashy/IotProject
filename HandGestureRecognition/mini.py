import tensorflow as tf
import numpy as np
import os
from PIL import Image

# 配置量化数据路径和模型路径
MODEL_FILE = "gesture_model_mobilenetv2.tflite"
QUANTIZATION_IMAGES_DIR = "quantization_images"
OUTPUT_MODEL_FILE = "gesture_model_mobilenetv2_int8.tflite"
IMG_SIZE = (128, 128)  # 与训练模型的输入尺寸一致


# 加载图片并预处理
def load_quantization_images(image_dir, img_size=IMG_SIZE):
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize(img_size)
        images.append(np.array(img) / 255.0)  # 标准化到 [0, 1]
    return np.array(images, dtype=np.float32)


# 量化的代表性数据生成函数
def representative_data_gen():
    images = load_quantization_images(QUANTIZATION_IMAGES_DIR)
    for img in images:
        yield [np.expand_dims(img, axis=0)]


# 读取 TFLite 模型并进行量化
def quantize_model(input_model_file, output_model_file):
    # 加载 TFLite 模型
    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_file) \
        if not input_model_file.endswith('.tflite') \
        else tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(input_model_file))

    # 配置为 INT8 量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # 输入为 uint8
    converter.inference_output_type = tf.uint8  # 输出为 uint8

    # 转换并保存量化模型
    tflite_quantized_model = converter.convert()
    with open(output_model_file, 'wb') as f:
        f.write(tflite_quantized_model)
    print(f"量化模型已保存到: {output_model_file}")


# 开始量化
quantize_model(MODEL_FILE, OUTPUT_MODEL_FILE)
