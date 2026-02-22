import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

pd.options.display.max_rows = 10  # displays 10 rows of a df usually first and last 5
pd.options.display.float_format = "{:.1f}".format  # 3.141532 to 3.1

rice_dataset_raw = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
)

rice_dataset = rice_dataset_raw[
    [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent",
        "Class",
    ]
]

# print(rice_dataset.describe()) #gives statistical description of dataset

# x_axis_data = "Area"
# y_axis_data = "Major_Axis_Length"
# z_axis_data = "Eccentricity"

# px.scatter_3d(
#     rice_dataset, x=x_axis_data, y=y_axis_data, z=z_axis_data, color="Class"
# ).show()

""" Normalizing the numerical Vlaue
    calculating Z-scores of each vlaues
"""

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes("number").columns
normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

normalized_dataset["Class"] = rice_dataset["Class"]

print(normalized_dataset.head())

# sets random for mutiple libraires
keras.utils.set_random_seed(42)

# Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
# then show 10 randomly selected rows.
# day2nd 
normalized_dataset["Class_Bool"] = (
    # Returns true if class is Cammeo, and false if class is Osmancik
    normalized_dataset["Class"]
    == "Cammeo"
).astype(int)
normalized_dataset.sample(10)
