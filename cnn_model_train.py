import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# === Config ===
DATA_FOLDER = 'gestures'
OUTPUT_FOLDER = 'data'
IMAGE_SIZE = 64
BATCH_SIZE = 30
EPOCHS = 50
TRAIN_SIZE = 26000
TEST_SIZE = 8000
VAL_SIZE = 8000

# === STEP 1: Load and preprocess gesture images ===
def preprocess_and_save_data():
    images = []
    labels = []

    for label_name in sorted(os.listdir(DATA_FOLDER)):
        class_folder = os.path.join(DATA_FOLDER, label_name)
        if not os.path.isdir(class_folder):
            continue

        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(img)
                labels.append(label_name)
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

    images = np.array(images, dtype=np.float32) / 255.0
    images = np.expand_dims(images, axis=-1)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save class labels
    with open(os.path.join(OUTPUT_FOLDER, 'label_classes.pkl'), 'wb') as f:
        pickle.dump(le.classes_, f)

    # Split the data according to specified sizes
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=42
    )
    
    # Ensure we have the exact sizes requested
    X_train, y_train = X_train[:TRAIN_SIZE], y_train[:TRAIN_SIZE]
    X_val, y_val = X_val[:VAL_SIZE], y_val[:VAL_SIZE]
    X_test, y_test = X_test[:TEST_SIZE], y_test[:TEST_SIZE]

    # Save datasets
    for name, data in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        with open(os.path.join(OUTPUT_FOLDER, f'{name}_images.pkl'), 'wb') as f: pickle.dump(data[0], f)
        with open(os.path.join(OUTPUT_FOLDER, f'{name}_labels.pkl'), 'wb') as f: pickle.dump(data[1], f)

    print(f"✅ Gesture images preprocessed and saved. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# === STEP 2: Load preprocessed data ===
def load_data():
    with open(os.path.join(OUTPUT_FOLDER, 'train_images.pkl'), 'rb') as f: train_images = pickle.load(f)
    with open(os.path.join(OUTPUT_FOLDER, 'train_labels.pkl'), 'rb') as f: train_labels = pickle.load(f)
    with open(os.path.join(OUTPUT_FOLDER, 'val_images.pkl'), 'rb') as f: val_images = pickle.load(f)
    with open(os.path.join(OUTPUT_FOLDER, 'val_labels.pkl'), 'rb') as f: val_labels = pickle.load(f)
    with open(os.path.join(OUTPUT_FOLDER, 'test_images.pkl'), 'rb') as f: test_images = pickle.load(f)
    with open(os.path.join(OUTPUT_FOLDER, 'test_labels.pkl'), 'rb') as f: test_labels = pickle.load(f)
    with open(os.path.join(OUTPUT_FOLDER, 'label_classes.pkl'), 'rb') as f: class_names = pickle.load(f)

    num_classes = len(class_names)

    # One-hot encode labels
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels, num_classes, class_names

# === STEP 3: Define CNN Model ===
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === STEP 4: Plot Training History ===
def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# === MAIN ===
if __name__ == '__main__':
    # Step 1: Only run this once unless you change the dataset
    if not os.path.exists(OUTPUT_FOLDER):
        preprocess_and_save_data()
    else:
        print("Using existing preprocessed data")

    # Step 2
    train_images, train_labels, val_images, val_labels, test_images, test_labels, num_classes, class_names = load_data()
    print(f"\nDataset Info:")
    print(f"Train samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Step 3
    model = create_model(num_classes)
    model.summary()

    # Step 4: Callbacks
    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    ]

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(train_images)

    # Train
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        steps_per_epoch=len(train_images) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        callbacks=callbacks
    )

    # Plot training history
    plot_history(history)

    # Final Evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")

    # Save the final model
    model.save('cnn_model_keras2.h5')
    print("Model saved as 'cnn_model_keras2.h5'")

    tf.keras.backend.clear_session()