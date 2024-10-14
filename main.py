import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to display the original image without resizing
def show_original_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Display the image without resizing
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Define your own CNN model
def create_custom_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function for object recognition using the custom CNN model
def recognize_objects_with_custom_cnn(image_path, model, top_k=5):
    # Load the VGG16 model with pre-trained ImageNet weights
    VGGmodel = VGG16(weights='imagenet')

    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the input shape expected by the custom CNN model
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for the custom CNN model
    img_array = preprocess_input(img_array)

    # Make predictions using the custom CNN model
    predictions = model.predict(img_array)

    # Make predictions using the VGG16 model
    classified = VGGmodel.predict(img_array)
    decoded_predictions = decode_predictions(classified, top=top_k)[0]

    # Display the original image without resizing
    show_original_image(image_path)

    # Display the predictions with confidence above a threshold
    print("Predictions:")
    for i, confidence in enumerate(predictions[0]):
        print(f"Class {i + 1}: Confidence = {confidence:.2f}")

    # Display the image with predicted class and probability
    plt.imshow(img)
    plt.axis('off')

    # Overlay the predicted class and probability on the image
    plt.text(10, 10, f"Top-{top_k} Predictions:", color='white', backgroundcolor='black')
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        plt.text(10, 30 + i * 15, f"{i + 1}: {label} ({score:.2f})", color='white', backgroundcolor='black')

    plt.show()

# Example usage
image_path = input("Select the image:")

# Create and train the custom CNN model (you may need to adjust parameters)
custom_model = create_custom_cnn_model()

# Object recognition using the custom CNN model
recognize_objects_with_custom_cnn(image_path, model=custom_model)
