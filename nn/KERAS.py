# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs) # 相當於上週 (Day 8) 自動建立上週的 Weight以及 bias的矩陣
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x) # activation上週是使用 sigmoid，依需做選擇也可使用 relu，But最後一層不能用，因為此例是用分類。

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer，上週使用的是 SGD
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),   # 上週使用的是 MSE
    # List of metrics to monitor
    # metrics=[keras.metrics.SparseCategoricalAccuracy()],    # 評估指標
)

model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()   # 這邊是 call function的方式 load x_data及 y_data，

# print(x_train.dtype) # check dtype

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,    # 資料量大的狀況下，可使用 batch_size來批次訓練
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    # validation_data=(x_val, y_val),    # 此例是邊 train 邊 test。例如 1 epochs之後，就進行一次 test來評估網路狀況。
) # 其實就是包裝過的 train step，內部自動化完成 Gradient descent與調整 weights and bias。 (forward與 back propagation)

print("History")
print(history.history)  

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128) 
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)