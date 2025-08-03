import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration
dataset_path = "gestures"
output_folder = "data"
image_size = 64
train_size = 26000
test_size = 8000
val_size = 8000

def load_and_preprocess_images():
    images = []
    labels = []
    
    # Get folder names (class names)
    folder_names = sorted([
        name for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    ])
    
    # Create label map
    label_map = {name: idx for idx, name in enumerate(folder_names)}
    print("Label mapping:", label_map)
    
    # Load and preprocess images
    for label_name in folder_names:
        folder_path = os.path.join(dataset_path, label_name)
        image_files = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
        
        for image_file in image_files:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype(np.float32) / 255.0
            label = label_map[label_name]
            
            images.append(img)
            labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
    )
    
    # Ensure exact sizes
    X_train, y_train = X_train[:train_size], y_train[:train_size]
    X_val, y_val = X_val[:val_size], y_val[:val_size]
    X_test, y_test = X_test[:test_size], y_test[:test_size]
    
    # Save datasets
    os.makedirs(output_folder, exist_ok=True)
    
    with open(os.path.join(output_folder, "train_images.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(output_folder, "train_labels.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(output_folder, "val_images.pkl"), "wb") as f:
        pickle.dump(X_val, f)
    with open(os.path.join(output_folder, "val_labels.pkl"), "wb") as f:
        pickle.dump(y_val, f)
    with open(os.path.join(output_folder, "test_images.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    with open(os.path.join(output_folder, "test_labels.pkl"), "wb") as f:
        pickle.dump(y_test, f)
    
    # Save label classes
    le = LabelEncoder()
    le.fit(labels)
    with open(os.path.join(output_folder, "label_classes.pkl"), "wb") as f:
        pickle.dump(le.classes_, f)
    
    print("\nDataset statistics:")
    print(f"Train images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Test images: {len(X_test)}")
    print(f"Total images: {len(X_train) + len(X_val) + len(X_test)}")
    print(f"Number of classes: {len(folder_names)}")
    print(f"Class names: {folder_names}")

if __name__ == "__main__":
    load_and_preprocess_images()