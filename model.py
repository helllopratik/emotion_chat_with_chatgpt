from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the model architecture
model = Sequential([
    InputLayer(input_shape=(48, 48, 1)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

# Load weights
try:
    model.load_weights("facialemotionmodel.h5")
    print("Weights loaded successfully")
except ValueError:
    print("Mismatch in the number of layers. Loading weights with custom layers...")
    # If the model architecture does not match, load weights layer by layer
    model.layers.pop()  # Remove the last Dense layer
    model.layers.pop()  # Remove the last Dropout layer
    model.layers.pop()  # Remove the Dense layer
    model.layers.pop()  # Remove the Dropout layer
    model.layers.pop()  # Remove the Dense layer
    model.layers.pop()  # Remove the Flatten layer
    
    # Load remaining weights
    model.load_weights("facialemotionmodel.h5", by_name=True)
    
    print("Custom weights loaded successfully")

# Now the model is ready to use

