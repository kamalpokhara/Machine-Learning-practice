import numpy as np
import pandas as pd
import plotly.express as px

# machine learning
import keras  # brings deeplearning libraires, keras is high lvl deeplearning API runs on top of tensorflow
import ml_edu.experiment  # helper modules
import ml_edu.results  # helper modules

# data visualization
import plotly.express as px  # library for interactive data visulation

chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
)


# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset.loc[
    :, ("TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE")
]

# print("Read dataset completed successfully.")
# print("Total number of rows: {0}\n\n".format(len(training_df.index)))
# print(training_df.head(200))

# print("Total number of rows: {0}\n\n".format(len(training_df.index)))
# print(training_df.describe(include="all"))

# # What is the maximum fare?
# max_fare = training_df["FARE"].max()
# print("What is the maximum fare? 				Answer: ${fare:.2f}".format(fare=max_fare))

# # What is the mean distance across all trips?
# mean_distance = training_df["TRIP_MILES"].mean()
# # print("What is the mean distance across all trips? 		Answer: {mean:.4f} miles".format(mean = mean_distance))
# print(f"What is the mean distance across all trips? 		Answer{mean_distance}")

# # How many cab companies are in the dataset?
# num_unique_companies = training_df["COMPANY"].nunique()
# # print("How many cab companies are in the dataset? 		Answer: {number}".format(number = num_unique_companies))
# print(f"How many cab companies are in the dataset? 		answer: {num_unique_companies}")

# # What is the most frequent payment type?
# most_freq_payment_type = training_df["PAYMENT_TYPE"].value_counts().idxmax()
# print(f"What is the most frequent payment type? 		Answer:{most_freq_payment_type}")

# # Are any features missing data?
# missing_values = training_df.isnull().sum().sum()
# # print("Are any features missing data? 				Answer:", "No" if missing_values == 0 else "Yes")
# print(f"Are there missing values? 				Answer:{'NO' if missing_values == 0 else 'Yes'} ")

##@title Code - View correlation matrix
print(training_df.corr(numeric_only=True))

# @title Code - View pairplot
#fig = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
#fig.show()