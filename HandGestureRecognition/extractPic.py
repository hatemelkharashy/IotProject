import os
import random
from PIL import Image
import numpy as np
import json
import shutil

# 设置目录路径
OUTPUT_DIR = "quantization_images"  # 保存提取的图片的目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ANN_DIR = "ann"
DATASET_DIR = "dataset"
GESTURES = ["fist", "like", "ok", "one", "palm"]


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


# 加载并预处理图像和标签数据
def load_data(annotations, img_size=(128, 128)):
    images = []
    labels = []
    label_map = {gesture: idx for idx, gesture in enumerate(GESTURES)}  # 映射手势到索引

    for file_path, label, bbox in annotations:
        if label not in label_map:
            continue  # 忽略未知标签
        img = Image.open(file_path).convert('RGB')
        img = crop_image(img, bbox)  # 裁剪图像
        img = img.resize(img_size)  # 调整大小
        img = np.array(img) / 255.0  # 归一化
        images.append(img)
        labels.append(label_map[label])

    return np.array(images), np.array(labels)


# 获取所有数据
annotations = load_annotations()
images, labels = load_data(annotations)

# 随机选择100张图片
num_images_to_select = 100
total_images = len(images)

# 如果数据集的图片数量超过100，均匀选择
if total_images > num_images_to_select:
    selected_indices = np.linspace(0, total_images - 1, num_images_to_select, dtype=int)
else:
    selected_indices = np.arange(total_images)

# 将选择的图片保存到指定目录
for i, idx in enumerate(selected_indices):
    img = images[idx]  # 获取图像数据
    img = (img * 255).astype(np.uint8)  # 反归一化
    img_pil = Image.fromarray(img)  # 转换为PIL图像
    filename = os.path.join(OUTPUT_DIR, f"image_{i + 1}.jpg")
    img_pil.save(filename)

print(f"{num_images_to_select} 张图片已保存到 {OUTPUT_DIR}")
