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

fare = (
    base_fare + (trip_miles * cost_per_mile) + (trip_minutes * cost_per_minute) + noise
)

df = pd.DataFrame(
    {"TRIP_MILES": trip_miles, "TRIP_MINUTES": trip_minutes, "FARE": fare}
)

print(df.head())
"""TRAIN MODEL with 1 Feature i.e trip_miles"""
x_train_1 = df["TRIP_MILES"].values.reshape(-1, 1)
y_train = df["FARE"].values

""""
 2. Build the Model
 Dense(1) means "1 Output Neuron".
 input_shape=(1,) means "we are feeding it 1 number (Miles)"
 defining simple keras model using sequntial API
"""
model_1 = keras.Sequential([layers.Dense(units=1, input_shape=(1,))])

"""Compile: learning rate 0.1 (step-size) loss=MSE"""

model_1.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001), loss="mean_squared_error"
)

print("Starting Training model 1...")
history_1 = model_1.fit(
    x_train_1, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2
)
print("Training Complete model 1")

"""Model 2- 2 features"""

x_train_2 = df[["TRIP_MILES", "TRIP_MINUTES"]].values
y_train = df["FARE"].values

model_2 = keras.Sequential([layers.Input(shape=(2,)), layers.Dense(units=1)])

model_2.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss="mean_squared_error"
)

print("Training model_2 ")
history_2 = model_2.fit(x_train_2, y_train, epochs=50, batch_size=32, verbose=0)
print("Training completed model 2")

final_loss_1 = history_1.history["loss"][-1]
final_loss_2 = history_2.history["loss"][-1]

print(f"\nFinal Cost (MSE) - 1 Feature:  {final_loss_1:.4f}")
print(f"Final Cost (MSE) - 2 Features: {final_loss_2:.4f}")

# plt.figure(figsize=(8, 5))
# plt.plot(history_1.history["loss"], label="model 1 (miles only)")
# plt.plot(history_2.history["loss"], label="model 2 (miles + time)")
# plt.title("Comparision: 1 Feature vs 2 Feature")
# plt.xlabel("Epochs")
# plt.ylabel("Cost (MSE)")
# plt.legend()
# plt.grid(True)
# plt.show()


""" Validation/Test """
# We use a specific random_state so we always get the same 10 trips
test_data = df.sample(10, random_state=99)

predictions_1 = model_1.predict(test_data[["TRIP_MILES"]])
predictions_2 = model_2.predict(test_data[["TRIP_MILES", "TRIP_MINUTES"]])

# comparision df
results = pd.DataFrame(
    {
        "Actual_Fare": test_data["FARE"],
        "Model_1_pred": predictions_1.flatten(),  # flatten()-> turns multi dimension arry to 1d array
        "Model_2_pred": predictions_2.flatten(),
    }
)

# difference inn error
results["Error_1"] = abs(results["Actual_Fare"] - results["Model_1_pred"])
results["Error_2"] = abs(results["Actual_Fare"] - results["Model_2_pred"])


print("\n-- Prediction Results first five --")
print(results.head().round(2))  # 45.33643 -> 45.34

indices = np.arange(10)  # gives numpy array [0,1,2,3,4,5,6,7,8,9]
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(
    indices - width / 2,
    results["Error_1"],
    width,
    label="Model 1 Error (Miles only)",
    color="red",
    alpha=0.6,
)
plt.bar(
    indices + width / 2,
    results["Error_2"],
    width,
    label="Model 2 Error (Miles + Time)",
    color="green",
    alpha=0.6,
)

plt.xlabel("Trip ID")
plt.ylabel("Error in dollars")
plt.title("Lower is better")
plt.xticks(indices, [f"Trip {i+3}" for i in indices])  # starts with trip 3
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
