import cv2 as cv
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the dataset directory
data_dir = r'C:\Users\Ali\footballer_classification\football_golden_foot\football_golden_foot'

# Folder names will be used as class names
class_names = os.listdir(data_dir)
label_dict = {i: class_name for i, class_name in enumerate(class_names)}

# Lists to store training and testing datasets
train_images, test_images = [], []
train_labels, test_labels = [], []

for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name)
    img_names = os.listdir(class_path)

    # Split 80% for training, 20% for testing
    random.shuffle(img_names)
    split_index = int(len(img_names) * 0.8)
    train_imgs = img_names[:split_index]
    test_imgs = img_names[split_index:]

    # Add images to the training dataset
    for img_name in train_imgs:
        img_path = os.path.join(class_path, img_name)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read in grayscale
        img = cv.resize(img, (100, 100))  # Resize to a fixed size
        train_images.append(img)
        train_labels.append(class_index)

    # Add images to the testing dataset
    for img_name in test_imgs:
        img_path = os.path.join(class_path, img_name)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read in grayscale
        img = cv.resize(img, (100, 100))  # Resize to a fixed size
        test_images.append(img)
        test_labels.append(class_index)

# Convert lists to NumPy arrays
train_images = np.array(train_images, dtype=np.uint8)
train_labels = np.array(train_labels, dtype=np.int32)
test_images = np.array(test_images, dtype=np.uint8)
test_labels = np.array(test_labels, dtype=np.int32)

print(f"Total number of classes: {len(class_names)}")
print(f"Training set: {len(train_images)} images")
print(f"Testing set: {len(test_images)} images")

# Create LBPH Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the model
face_recognizer.train(train_images, train_labels)

# Save the trained model
model_path = "footballer_recognizer.yml"
face_recognizer.save(model_path)
print("Model successfully saved.")

# Load the model
face_recognizer.read(model_path)

# Function to classify a single image
def classify_image(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read in grayscale
    img = cv.resize(img, (100, 100))  # Resize to a fixed size
    label, confidence = face_recognizer.predict(img)
    
    print(f'Predicted: {label_dict[label]}, Confidence: {confidence:.2f}')
    return label_dict[label]


y_true = []
y_pred = []

# Predict each image in the test set
for i, test_img in enumerate(test_images):
    label, confidence = face_recognizer.predict(test_img)
    y_true.append(test_labels[i])
    y_pred.append(label)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Example classification
classify_image('C:/Users/Ali/footballer_classification/messi.jpg')

# Individual test predictions
'''
face_recognizer.read("footballer_recognizer.yml")

y_true = []
y_pred = []

print("\nPredictions on the test dataset:")

for i, test_img in enumerate(test_images):
    label, confidence = face_recognizer.predict(test_img)
    
    real_class = label_dict[test_labels[i]]
    predicted_class = label_dict[label]
    
    print(f'Actual: {real_class} | Predicted: {predicted_class} | Confidence: {confidence:.2f}')
    
    y_true.append(test_labels[i])
    y_pred.append(label)

# Calculate accuracy again
accuracy = accuracy_score(y_true, y_pred)
print(f"\nModel accuracy: {accuracy * 100:.2f}%")
'''




















