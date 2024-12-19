import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# 加载 TFLite 模型
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter


# 运行预测
def predict(image, interpreter, input_size=(128, 128)):
    # 获取模型的输入和输出张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 预处理图像
    img_resized = image.resize(input_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度

    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # 执行推理
    interpreter.invoke()

    # 获取输出张量并找到预测类别
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data, axis=1)[0]
    confidence = output_data[0][predicted_label]
    return predicted_label, confidence


# 标注手势
def annotate_image(image, bbox, label, confidence, label_map):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # 使用默认字体

    # 设置框和文本
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
    label_text = f"{label_map[label]}: {confidence:.2f}"
    draw.text((x, y - 10), label_text, fill="green", font=font)

    return image


# 主函数
def main(image_path, tflite_model_path):
    # 加载类别标签映射
    label_map = {0: "fist", 1: "like", 2: "ok", 3: "one", 4: "palm", 5: "no_gesture"}

    # 加载图像
    original_image = Image.open(image_path).convert("RGB")
    img_width, img_height = original_image.size

    # 加载 TFLite 模型
    interpreter = load_tflite_model(tflite_model_path)

    # 预测手势类别
    predicted_label, confidence = predict(original_image, interpreter)

    # 假设整个图像包含手势并创建一个边框标注
    bbox = (10, 10, img_width - 20, img_height - 20)  # 示例，适应整个图像范围
    annotated_image = annotate_image(original_image, bbox, predicted_label, confidence, label_map)

    # 显示标注后的图像
    annotated_image.show()


# 运行代码示例
image_path = "test_fist.jpg"  # 替换为待识别的图像路径
tflite_model_path = "gesture_model_mobilenetv2.tflite"  # 替换为已保存的 TFLite 模型路径
main(image_path, tflite_model_path)
