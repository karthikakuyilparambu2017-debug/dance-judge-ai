import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# ✅ Set dataset directory
dataset_dir = "C:/Users/DELL/dance_judge_AI/dataset"

# ✅ Image augmentation and data preparation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# ✅ Training and validation data generators
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',   # ✅ Change from 'binary' to 'categorical'
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',   # ✅ Change to 'categorical'
    subset='validation'
)

# ✅ Model architecture with SOFTMAX activation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # ✅ Use SOFTMAX for multi-class classification
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Model checkpoint callback (saving in .h5 format)
model_checkpoint = ModelCheckpoint(
    "C:/Users/DELL/dance_judge_AI/costume_model_v3.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ✅ Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[model_checkpoint]
)

# ✅ Save the final model in .h5 format
model.save("C:/Users/DELL/dance_judge_AI/costume_model_v3.h5")
print("✅ Model saved successfully as costume_model_v3.h5!")

# ✅ Plot training accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # ✅ Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # ✅ Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

# ✅ Display the training history plot
plot_training_history(history)
