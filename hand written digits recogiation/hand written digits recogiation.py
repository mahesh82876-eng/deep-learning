import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
tf.keras.datasets.mnist.load_data()

#load the data from the tensor flow
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
#now we should see how the data looks like \
'''
print("x trainig image:",x_train)
print("y_training image:",y_train)
print("x_testing image:",x_test)
print("y testing image:",y_test)
'''
'''
plt.imshow(x_train[1],cmap="gray")
plt.title("f:lable:{y_train[0]}")
plt.axis("off")
plt.show()
'''

#normalising the image
x_train=x_train/255.0
Y_train=y_train/255.0
#building the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation="relu"),

    keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
# ==========================================
# 7. Callbacks
# ==========================================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    save_best_only=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
# ==========================================
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ==========================================
# 9. Evaluate Model
# ==========================================
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)