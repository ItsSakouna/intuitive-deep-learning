"""
Author: Isaiah Kongmanivong
Date: December 10, 2025
Course: CIS-2532-NET01
Assignment: III - Convolutional Neural Network Implementation

Original Code Credits:
Author: Joseph Lee Wei En
Source: https://github.com/josephlee94/intuitive-deep-learning
Repository: intuitive-deep-learning/Part 2: Image Recognition CIFAR-10
Tutorial: https://medium.com/intuitive-deep-learning/build-your-first-convolutional-neural-network-to-recognize-images-84b9c78fe0ce

Description:
This code has been duplicated, tested, and verified as part of an academic assignment.
The program builds a Convolutional Neural Network (CNN) using Keras to recognize
and classify images from the CIFAR-10 dataset into 10 different categories.

Modifications:
- Added comprehensive documentation and attribution
- Updated for modern TensorFlow/Keras compatibility
- Added detailed comments for educational purposes
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model

np.random.seed(42)

print("="*70)
print("CIFAR-10 IMAGE CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORKS")
print("="*70)
print()

print("Step 1: Loading CIFAR-10 dataset...")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"Dataset loaded successfully!")
print(f"  Training images: {X_train.shape}")
print(f"  Training labels: {y_train.shape}")
print(f"  Test images: {X_test.shape}")
print(f"  Test labels: {y_test.shape}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\nDataset contains {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print("\nStep 2: Visualizing sample images from the dataset...")

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i][0]])

plt.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('cifar10_sample_images.png', dpi=150, bbox_inches='tight')
print("Sample images saved as 'cifar10_sample_images.png'")
plt.show()

print("\nStep 3: Preprocessing the data...")

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("Images normalized (pixel values: 0-1)")

y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

print("Labels converted to one-hot encoding")
print(f"  Training labels shape: {y_train_categorical.shape}")
print(f"  Test labels shape: {y_test_categorical.shape}")

print("\nStep 4: Building the Convolutional Neural Network architecture...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', 
           input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

print("CNN architecture created successfully!")
print("\nModel Summary:")
print("-" * 70)
model.summary()
print("-" * 70)

print("\nStep 5: Compiling the model...")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")
print("  Optimizer: Adam")
print("  Loss Function: Categorical Crossentropy")
print("  Metrics: Accuracy")

print("\nStep 6: Training the model...")
print("Note: This will take several minutes (or longer without GPU)")
print("-" * 70)

history = model.fit(
    X_train, y_train_categorical,
    batch_size=128,
    epochs=50,
    validation_data=(X_test, y_test_categorical),
    verbose=1
)

print("-" * 70)
print("✓ Training completed successfully!")

print("\nStep 7: Evaluating the model on test set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

print(f"\n{'='*70}")
print("FINAL TEST RESULTS:")
print(f"{'='*70}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*70}")
print("\nStep 8: Creating training history visualizations...")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_training_history.png', dpi=150, bbox_inches='tight')
print("Training history saved as 'cifar10_training_history.png'")
plt.show()

print("\nStep 9: Saving the trained model...")

model.save('cifar10_cnn_model.h5')
print("Model saved as 'cifar10_cnn_model.h5'")
print("\nStep 10: Making predictions on sample test images...")

predictions = model.predict(X_test[:10])
predicted_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    
    predicted_label = class_names[predicted_classes[i]]
    true_label = class_names[y_test[i][0]]
    confidence = predictions[i][predicted_classes[i]] * 100
    
    color = 'green' if predicted_classes[i] == y_test[i][0] else 'red'
    
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}\n({confidence:.1f}%)", 
               color=color, fontsize=9)

plt.suptitle('Sample Predictions on Test Images', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('cifar10_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions saved as 'cifar10_predictions.png'")
plt.show()

print("\nDetailed predictions for first 10 test images:")
print("-" * 70)
for i in range(10):
    predicted_label = class_names[predicted_classes[i]]
    true_label = class_names[y_test[i][0]]
    confidence = predictions[i][predicted_classes[i]] * 100
    correct = "✓" if predicted_classes[i] == y_test[i][0] else "✗"
    
    print(f"{correct} Image {i+1}: Predicted={predicted_label:12s} "
          f"(conf: {confidence:5.1f}%) | True={true_label}")
print("-" * 70)

print("\nStep 11: Generating confusion matrix...")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

all_predictions = model.predict(X_test)
all_predicted_classes = np.argmax(all_predictions, axis=1)
all_true_classes = y_test.flatten()

cm = confusion_matrix(all_true_classes, all_predicted_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - CIFAR-10 Classification', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('cifar10_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved as 'cifar10_confusion_matrix.png'")
plt.show()

print("\nClassification Report:")
print("="*70)
print(classification_report(all_true_classes, all_predicted_classes, 
                          target_names=class_names))
print("="*70)

print("\n" + "="*70)
print("PROGRAM COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  1. cifar10_sample_images.png - Sample training images")
print("  2. cifar10_training_history.png - Training/validation curves")
print("  3. cifar10_predictions.png - Sample predictions visualization")
print("  4. cifar10_confusion_matrix.png - Confusion matrix heatmap")
print("  5. cifar10_cnn_model.h5 - Trained model file")

print("\nModel Performance Summary:")
print(f"  Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  Final Test Loss: {test_loss:.4f}")

print("\nNext Steps:")
print("  1. Upload this file to your GitHub repository")
print("  2. Ensure all generated images and model files are included")
print("  3. Maintain proper attribution to the original author")
print("  4. Submit your GitHub URL in your assignment document")
print("="*70)

print("\nAll tasks completed successfully!")
print("\nTo load the saved model later, use:")
print("  from keras.models import load_model")
print("  model = load_model('cifar10_cnn_model.h5')")
print("="*70)
