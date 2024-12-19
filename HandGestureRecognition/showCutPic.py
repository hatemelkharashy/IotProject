import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义路径
ANN_DIR = "ann"
DATASET_DIR = "dataset"
GESTURES = ["peace", "rock", "ok", "stop", "fist"]


# 解析 JSON 文件函数
def load_annotations():
    annotations = []
    for gesture in GESTURES:
        ann_path = os.path.join(ANN_DIR, f"{gesture}.json")
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
            for image_id, data in ann_data.items():
                file_path = os.path.join(DATASET_DIR, f"train_val_{gesture}", f"{image_id}.jpg")
                if os.path.exists(file_path):
                    # 提取边界框和标签
                    for bbox, label in zip(data["bboxes"], data["labels"]):
                        if label != "no_gesture":  # 忽略无手势的标签
                            annotations.append((file_path, label, bbox))
    return annotations


# 裁剪图像函数
def crop_image(img, bbox):
    width, height = img.size
    x, y, w, h = bbox
    left = int(x * width)
    top = int(y * height)
    right = left + int(w * width)
    bottom = top + int(h * height)
    return img.crop((left, top, right, bottom))


# 加载并裁剪数据
def load_and_crop_images(annotations, img_size=(96, 96)):
    cropped_images = []
    labels = []

    for file_path, label, bbox in annotations:
        img = Image.open(file_path).convert('RGB')
        cropped_img = crop_image(img, bbox)  # 裁剪图像
        cropped_img = cropped_img.resize(img_size)  # 调整大小
        cropped_images.append((np.array(cropped_img), label))

    return cropped_images


# 展示裁剪后的样例图像
def show_cropped_samples(cropped_images, max_samples=9):
    plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(cropped_images[:max_samples]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 主逻辑
if __name__ == "__main__":
    # 加载注释
    annotations = load_annotations()

    # 裁剪并加载图片
    cropped_images = load_and_crop_images(annotations)

    # 可视化裁剪结果
    show_cropped_samples(cropped_images)
