import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import pickle
import time

# Load the trained model and class names
model = load_model("cnn_model_keras2.h5")
with open('data/label_classes.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Image dimensions (must match training)
IMAGE_SIZE = 64

# Initialize text-to-speech engine
engine = pyttsx3.init()
last_prediction_time = 0
prediction_cooldown = 2  # seconds
current_prediction = ""
prediction_history = []

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence

def speak_prediction(prediction):
    engine.say(prediction)
    engine.runAndWait()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    
    # Define ROI (Region of Interest)
    x1, y1 = int(width * 0.25), int(height * 0.1)
    x2, y2 = x1 + 300, y1 + 300
    roi = frame[y1:y2, x1:x2]
    
    # Display ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Get current time
    current_time = time.time()
    
    # Make prediction every 'prediction_cooldown' seconds
    if current_time - last_prediction_time > prediction_cooldown:
        try:
            prediction, confidence = predict_gesture(roi)
            if confidence > 0.8:  # Only accept predictions with high confidence
                current_prediction = prediction
                prediction_history.append(current_prediction)
                if len(prediction_history) > 5:
                    prediction_history.pop(0)
                last_prediction_time = current_time
                speak_prediction(current_prediction)
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # Display prediction
    cv2.putText(frame, f"Prediction: {current_prediction}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display confidence
    cv2.putText(frame, "Put your hand in the green box", (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Sign Language Interpreter", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()