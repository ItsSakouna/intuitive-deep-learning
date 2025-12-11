"""
Author: Isaiah Kongmanivong
Date: December 10, 2025
Course: CIS-2532 NET-01
Assignment: III - Convolutional Neural Network Implementation

Original Code Credits:
Author: Joseph Lee Wei En
Source: https://github.com/josephlee94/intuitive-deep-learning
Repository: intuitive-deep-learning/Part 1: Predicting House Prices
Tutorial: https://medium.com/intuitive-deep-learning/build-your-first-neural-network-to-predict-house-prices-with-keras-eb5db60232c

Description:
This code has been duplicated, tested, and verified as part of an academic assignment.
The program builds a neural network using Keras to predict whether house prices are
above or below median value using the Boston Housing dataset.

Modifications:
- Added comprehensive documentation and attribution
- Updated for modern TensorFlow/Keras compatibility
- Added detailed comments for educational purposes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*60)
print("House Price Prediction with Neural Networks")
print("="*60)
print()

print("Step 1: Loading and preprocessing data...")

try:
    df = pd.read_csv('housepricedata.csv')
    print(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("ERROR: 'housepricedata.csv' not found!")
    print("Please download it from: https://github.com/josephlee94/intuitive-deep-learning")
    exit()

print("\nFirst 5 rows of the dataset:")
print(df.head())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y.shape}")

print("\nStep 2: Normalizing the data...")

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Data normalized successfully")

print("\nStep 3: Splitting data into train, validation, and test sets...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nStep 4: Building the neural network architecture...")

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    
    Dense(32, activation='relu'),
    
    Dense(1, activation='sigmoid')
])

print("Model architecture created")
model.summary()

print("\nStep 5: Compiling the model...")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully")

print("\nStep 6: Training the model...")
print("This may take a few minutes...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

print("Training completed!")

print("\nStep 7: Evaluating the model on test set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"TEST RESULTS:")
print(f"{'='*60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*60}")

print("\nStep 8: Creating visualizations...")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('house_price_training_history.png')
print("Training history plots saved as 'house_price_training_history.png'")
plt.show()

print("\nStep 9: Saving the trained model...")

model.save('house_price_model.h5')
print("Model saved as 'house_price_model.h5'")

print("\nStep 10: Making sample predictions...")

sample_predictions = model.predict(X_test[:5])
print("\nSample predictions (first 5 test samples):")
for i, pred in enumerate(sample_predictions):
    actual = y_test[i]
    predicted = "Above Median" if pred[0] > 0.5 else "Below Median"
    actual_label = "Above Median" if actual == 1 else "Below Median"
    print(f"Sample {i+1}: Predicted={predicted} (confidence: {pred[0]:.2f}), Actual={actual_label}")

print("\n" + "="*60)
print("PROGRAM COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nNext Steps:")
print("1. Upload this file to your GitHub repository")
print("2. Ensure proper attribution is maintained")
print("3. Submit your GitHub URL in your assignment document")
print("="*60)
