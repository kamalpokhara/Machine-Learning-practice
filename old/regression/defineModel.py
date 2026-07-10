# 1. Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
n_sample = 1000  # simple variable assign

trip_miles = np.random.uniform(1, 30, n_sample)
trip_minutes = trip_miles * 3 + np.random.normal(0, 5, n_sample)
base_fare = 2.0
cost_per_mile = 1.25
cost_per_minute = 0.25
noise = np.random.normal(0, 2, n_sample)

fare = base_fare + (trip_miles * cost_per_mile) + (trip_minutes * cost_per_mile) + noise

df = pd.DataFrame(
    {"TRIP_MILES": trip_miles, "TRIP_MINUTES": trip_minutes, "FARE": fare}
)

# print("Show first five rows of data;")
# print(df.head())  #show first five rows

# code for graph

# plt.figure(figsize=(8, 6))  # canvas size 8*6
# sns.scatterplot(x="TRIP_MILES", y="FARE", data=df)
# plt.title("Trip miles VS Fare")
# plt.xlabel("Trip miles")
# plt.ylabel("Fare")
# plt.show()

"""TRAIN MODEL with 1 Feature i.e trip_miles"""
x_train_1 = df["TRIP_MILES"]
y_train = df["FARE"]

"""" 
 2. Build the Model
 Dense(1) means "1 Output Neuron". 
 input_shape=(1,) means "we are feeding it 1 number (Miles)"
"""
# defining simple keras model using sequntial API
model_1 = keras.Sequential([layers.Dense(units=1, input_shape=(1,))])

"""Compile: learning rate 0.1 (step-size) loss=MSE"""

model_1.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001), loss="mean_squared_error"
)

"""Training the model"""
print("Starting Training...")
history_1 = model_1.fit(
    x_train_1, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2
)
print("Training Complete")

plt.figure(figsize=(8, 5))
plt.plot(history_1.history["loss"], label="Train Loss")
plt.plot(
    history_1.history["val_loss"], label="Validation Loss"
)  # If you used validation_split
plt.title("Model 1: Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
