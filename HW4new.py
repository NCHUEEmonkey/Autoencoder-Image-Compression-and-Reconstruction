import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Set the directory where your images are located
directory = 'images/images/n02085620-Chihuahua'

# Load all images into a list
all_images = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(directory, filename)
        try:
            img = image.load_img(img_path, target_size=(128, 128))
            img_data = image.img_to_array(img) / 255.0
            all_images.append(img_data)
        except Exception as e:
            print(f"Error loading '{filename}': {e}")
            continue

# Convert the list of images to a numpy array
all_images = np.array(all_images)

# Define the autoencoder model
input_img = Input(shape=(128, 128, 3))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # Add padding='same' here
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(all_images, all_images,
                epochs=500,
                batch_size=32,
                shuffle=True)

# Use the autoencoder to reconstruct images
reconstructed_images = autoencoder.predict(all_images)

# Display and save only five original, reconstructed, and latent space representations
for i, img_data in enumerate(all_images[:5]):
    # Display the images and latent space representation
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img_data)
    plt.axis('off')

    # Latent Space Representation
    plt.subplot(1, 3, 2)
    encoder = Model(input_img, encoded)
    latent_representation = encoder.predict(np.expand_dims(img_data, axis=0))
    latent_img = np.mean(latent_representation, axis=-1)
    latent_img = np.squeeze(latent_img)
    plt.title('Latent Space')
    plt.imshow(latent_img)
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(1, 3, 3)
    plt.title('Reconstructed Image')
    reconstructed_img = reconstructed_images[i]
    plt.imshow(reconstructed_img)
    plt.axis('off')

    plt.show()
