# Import necessary modules from Keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load CIFAR10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess input data
# X_train = X_train
# X_test = X_test
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the input tensor shape
input_tensor = Input(shape=(32, 32, 3))

# Load the VGG16 network, ensuring the head FC layer sets are left off
vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)

# Construct the head of the model that will be placed on top of the base model
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(512, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation="softmax"))

# Combine the base model with the new FC layers
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# Loop over all layers in the base model and freeze them so they will not be updated during training
for layer in model.layers[:19]:
    layer.trainable = False

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=["accuracy"],
)

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=20)

# Evaluate the model on test data
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("my_model.h5")
