import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# In retrospect, this is not AT ALL how diffusion is supposed to work
# It was supposed to be a 'from first principles' type of thing :(

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_images = train_images[train_labels == 5]

def build_denoiser():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    return model


# visualization
def visualize_denoising(denoiser, noisy_images, num_samples=10):
    predictions = denoiser.predict(noisy_images[:num_samples])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    
    for ax, img in zip(axes[0], noisy_images):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title("Noisy")
    
    for ax, img in zip(axes[1], predictions):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title("Denoised")
    
    plt.show()



denoiser = build_denoiser()

# make the denoisers
num_denoisers = 10
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
denoisers = [build_denoiser() for _ in range(num_denoisers)]
for denoiser in denoisers:
     denoiser.compile(optimizer=optimizer, loss="mse")

# parameters
num_epochs = 5
batch_size = 16
input_images = np.random.normal(size=train_images.shape)
noise_scale =  0.35

for denoiser_num, denoiser in enumerate(denoisers):

    

    print(f"\nTraining denoiser {denoiser_num + 1}/{num_denoisers}...\n")
    
    # corrupt the images
    noisy_images = np.clip(input_images + noise_scale * np.random.normal(size=train_images.shape), 0, 1)

    # create dataset to do random shuffling
    train_dataset = tf.data.Dataset.from_tensor_slices((noisy_images, train_images))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(num_epochs):
        for step, (batch_noisy_images, batch_train_images) in enumerate(train_dataset):
            loss = denoiser.train_on_batch(batch_noisy_images, batch_train_images)
            
            if step % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}, Loss: {loss:.4f}")
        
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")
    
    visualize_denoising(denoiser, noisy_images)

    input_images = denoiser.predict(noisy_images)

print("\nTraining completed.")
