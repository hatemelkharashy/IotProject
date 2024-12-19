import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 手势类别
GESTURES = ["one", "peace", "stop", "ok", "rock", "fist"]
INTPUT_SHAPE = 96


# 加载 TFLite 模型
def load_tflite_model(model_path="gesture_model_gray_int8.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


# 图像预处理（灰度图）
def preprocess_frame(frame, input_size=(INTPUT_SHAPE, INTPUT_SHAPE)):
    # 转换为灰度图并调整大小
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray).resize(input_size)
    img = np.array(img, dtype=np.uint8)
    # 添加通道维度
    img = np.expand_dims(img, axis=-1)
    # 添加批次维度
    img = np.expand_dims(img, axis=0)
    return img


# 模型推理
def predict_gesture(interpreter, input_details, output_details, frame):
    input_data = preprocess_frame(frame, tuple(input_details[0]['shape'][1:3]))

    # 获取量化的输入信息（如果有的话）
    input_scale, input_zero_point = input_details[0]['quantization']

    # 手动量化输入
    input_data = input_data.astype(np.float32) / 255.0
    input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)

    # 设置输入数据
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 获取量化的输出信息（如果有的话）
    output_scale, output_zero_point = output_details[0]['quantization']

    # 获取输出数据并去量化
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    # 获取预测结果
    gesture_idx = np.argmax(output_data)
    confidence = output_data[0][gesture_idx]

    # 只有当预测概率大于50%时才输出预测结果，否则输出 "no_gesture"
    if confidence > 0.5:
        return GESTURES[gesture_idx], confidence
    else:
        return "no_gesture", confidence


# 初始化摄像头
def start_camera():
    interpreter, input_details, output_details = load_tflite_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        # 获取手势预测
        gesture, confidence = predict_gesture(interpreter, input_details, output_details, frame)

        # 在视频帧上显示预测结果
        text = f"{gesture} ({confidence * 100:.2f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示视频流
        cv2.imshow("Gesture Recognition", frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
