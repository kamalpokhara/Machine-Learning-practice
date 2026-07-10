"""Simple Linear Regression Model"""

import numpy as np
import pandas as py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# synthetic fata
x = np.linspace(0, 10, 100)
y = 3 * x + 5 + np.random.normal(0, 2, 100)

"""Build Model"""

model = keras.Sequential([
    layers.Input(shape=(1)), 
    layers.Dense(units=1)
])

model.compile(tf.keras.optimizers.SGD(learning_rate = 0.01), loss='mse' )

"""Train Model"""
print("Training started...")
model.fit(x,y,epochs = 20, batch_size = 32, verbose = 1)
print("Training completed")

"""Prediction"""
y_pred = model.predict(x)

plt.scatter(x, y, label='data')

