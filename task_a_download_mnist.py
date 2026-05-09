# task_a_download_mnist.py
# This script ONLY downloads and loads the MNIST dataset

import tensorflow as tf
from tensorflow import keras

print("=== Task (a): Downloading MNIST Dataset ===")

# Download and load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Display dataset information
print("\n✅ MNIST Dataset successfully downloaded and loaded!")
print(f"\nDataset Structure:")
print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"\nImage details:")
print(f"  - Image size: 28x28 pixels")
print(f"  - Number of training images: 60,000")
print(f"  - Number of test images: 10,000")
print(f"  - Digits: 0 through 9")
print(f"  - Pixel value range: {x_train.min()} to {x_train.max()}")

# Verify data is accessible
print(f"\n✅ First training image label: {y_train[0]}")
print(f"✅ First test image label: {y_test[0]}")
