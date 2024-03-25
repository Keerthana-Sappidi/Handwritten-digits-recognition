import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

# Save the model
model.save('handwritten_model.h5')

print("Model saved successfully!")

# Load the model
model = tf.keras.models.load_model('handwritten_model.h5')

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)

print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Load a PNG image
image_paths = [
    r'C:\Users\supriya\OneDrive\Desktop\digit3.png',
    r'C:\Users\supriya\OneDrive\Desktop\digit2.png',
    r'C:\Users\supriya\OneDrive\Desktop\digit4.png'
]

for image_path in image_paths:
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    img = cv2.resize(img, (28, 28))
    img = np.invert(img)
    img = img.reshape(1, 28, 28)  # Reshape to match model input shape
    img = tf.keras.utils.normalize(img, axis=1)

    # Make prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    # Visualize the image and predicted label
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.show()
