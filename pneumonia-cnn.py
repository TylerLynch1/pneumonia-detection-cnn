from pyspark.sql import SparkSession
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("PneumoniaDetectionCNN") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Define dataset paths
train_dir = "/chest-xray/train"
val_dir = "/chest-xray/val"
test_dir = "/chest-xray/test"

# CNN model parameters
input_shape = (150, 150, 1)
num_classes = 2
batch_size = 32
epochs = 3

# Function to preprocess and load images
def load_images(data_dir, target_size, color_mode, batch_size, shuffle):
    """
    Preprocess and load images from a given directory.

    Args:
        data_dir (str): Path to the directory containing the dataset.
        target_size (tuple): Target size to resize images (height, width).
        color_mode (str): Color mode for image loading ('grayscale' or 'rgb').
        batch_size (int): Number of images per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        tensorflow.keras.preprocessing.image.DirectoryIterator: An iterator for the dataset.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode=color_mode,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=shuffle
    )

# Load datasets
train_gen = load_images(train_dir, input_shape[:2], "grayscale", batch_size, True)
val_gen = load_images(val_dir, input_shape[:2], "grayscale", batch_size // 2, False)
test_gen = load_images(test_dir, input_shape[:2], "grayscale", batch_size, False)

# Define the CNN model
def build_cnn(input_shape, num_classes):
    """
    Build a Convolutional Neural Network (CNN) model for image classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: The compiled CNN model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

# Build and compile the model
model = build_cnn(input_shape, num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1
)

# Evaluate the model
print("Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Stop the Spark session
spark.stop()
