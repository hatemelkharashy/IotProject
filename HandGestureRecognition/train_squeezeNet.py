import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

# Define paths and gesture labels
ANN_DIR = "ann"
DATASET_DIR = "dataset"
GESTURES = ["fist", "like", "ok", "one", "palm"]


# Load annotations from JSON files
def load_annotations():
    annotations = []
    for gesture in GESTURES:
        ann_path = os.path.join(ANN_DIR, f"{gesture}.json")
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
            for image_id, data in ann_data.items():
                file_path = os.path.join(DATASET_DIR, f"train_val_{gesture}", f"{image_id}.jpg")
                if os.path.exists(file_path):
                    # Use the first label that is not "no_gesture"
                    label = next((lbl for lbl in data["labels"] if lbl != "no_gesture"), "no_gesture")
                    annotations.append((file_path, label))
    return annotations


# Load and preprocess images and labels
def load_data(annotations, img_size=(128, 128)):
    images = []
    labels = []
    label_map = {gesture: idx for idx, gesture in enumerate(GESTURES)}  # Map gestures to indices

    for file_path, label in annotations:
        if label not in label_map:
            continue  # Skip unknown labels
        img = Image.open(file_path).convert('RGB')
        img = img.resize(img_size)
        img = np.array(img) / 255.0  # Normalize
        images.append(img)
        labels.append(label_map[label])

    return np.array(images), np.array(labels)


# Load data
annotations = load_annotations()
images, labels = load_data(annotations)

# Split data into training and validation sets
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
train_size = int(0.8 * len(images))
val_size = len(images) - train_size
train_dataset = dataset.take(train_size).shuffle(1000).batch(32).repeat()
val_dataset = dataset.skip(train_size).batch(32)


# SqueezeNet implementation
def FireModule(x, squeeze_channels, expand_channels):
    x = layers.Conv2D(squeeze_channels, (1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    expand1x1 = layers.Conv2D(expand_channels, (1, 1), activation='relu')(x)
    expand3x3 = layers.Conv2D(expand_channels, (3, 3), padding='same', activation='relu')(x)
    x = layers.concatenate([expand1x1, expand3x3])
    return x


def build_squeezenet(input_shape=(128, 128, 3), num_classes=len(GESTURES)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = FireModule(x, squeeze_channels=16, expand_channels=64)
    x = FireModule(x, squeeze_channels=16, expand_channels=64)
    x = FireModule(x, squeeze_channels=32, expand_channels=128)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = FireModule(x, squeeze_channels=32, expand_channels=128)
    x = FireModule(x, squeeze_channels=48, expand_channels=192)
    x = FireModule(x, squeeze_channels=48, expand_channels=192)
    x = FireModule(x, squeeze_channels=64, expand_channels=256)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = FireModule(x, squeeze_channels=64, expand_channels=256)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax')(x)

    model = models.Model(inputs, outputs)
    return model


model = build_squeezenet()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    steps_per_epoch=train_size // 32,
    validation_steps=val_size // 32
)


# Save model as TFLite format
def save_model_as_tflite(model, filename="gesture_model_squeezenet.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
    tflite_model = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model)


save_model_as_tflite(model)
print("Model saved as gesture_model_squeezenet.tflite")
