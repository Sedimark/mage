from tensorflow import keras

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Define the model
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(9,)))
model.add(Dense(128, activation='relu')),
model.add(Dense(128, activation='relu')),
model.add(Dense(64, activation='relu')),
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Save the model
model.save("model.keras")

print("Model saved as model.keras")
