# Load
import keras
import ml_edu.experiment 
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

pd.options.display.max_rows = 10  # displays 10 rows of a df usually first and last 5
pd.options.display.float_format = "{:.1f}".format  # 3.141532 to 3.1

print("Loaded the libraries and set max rows 10 and float format to 1 decimal place")

rice_dataset_raw = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
)
print("rice dataset loaded")

#loading specified columns 
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
    ]]

# print(rice_dataset.describe()) #gives statistical description of dataset
# x_axis_data = "Area"
# y_axis_data = "Major_Axis_Length"
# z_axis_data = "Eccentricity"

# px.scatter_3d(
#     rice_dataset, x=x_axis_data, y=y_axis_data, z=z_axis_data, color="Class"
# ).show()

# Normalizing the numerical Vlaue
# calculating Z-scores of each vlaues 
# Z = (X - mean) / std

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes("number").columns
normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

normalized_dataset["Class"] = rice_dataset["Class"]

# print(normalized_dataset.head())

# sets random for mutiple libraires
keras.utils.set_random_seed(42)

# Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
# then show 10 randomly selected rows.

'''April 26 2026'''
# return true for Cammeo, false for osmanic 
normalized_dataset["Class_Bool"] = (
    normalized_dataset["Class"] == "Cammeo"
).astype(int)
print("Normalised datased: \n",normalized_dataset.sample(10))

#train validation and test splits of 80 10 10
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th +round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac = 1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]

validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

print("test data head: \n",test_data.head())

'''APRIL 27 '''
#prevent label leakage
label_columns = ["Class", "Class_Bool"]

train_features= train_data.drop(columns =label_columns)
train_labels= train_data['Class_Bool'].to_numpy()

validation_features = validation_data.drop(columns = label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()

test_features = test_data.drop(columns = label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

#traing model on Eccentricity and Major axis length and area

input_features = [
    'Eccentricity',
    'Major_Axis_length',
    'Area',
]

#define the functions
def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],

    )-> keras.Model:
    '''Create and complie simple calssifi model'''
    model_inputs = [
        keras.Input(name=feature, shape=(1,))
        for feature in  settings.input_features
    ]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)

    model_output = keras.layers.Dense(
        units=1, name = 'dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)

    model = keras.Model(inputs = model_inputs, outputs = model_output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        #binary_crossentorpy standard loos fun for binary classification
        loss=keras.losses.binary_crossentropy(),
        metrics=metrics,
    )
    return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset:pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings, 
    
    )-> ml_edu.experiment.Experiment:
    '''feed a ds into the model to train''' 

    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }
    history = model.fit(
        x=features,
        y= labels,
        batch_size= settings.batch_size,
        epochs= settings.number_epochs,
    )

    return ml_edu.experiment.Experiment(
        name = experiment_name,
        settings= settings,
        model=model,
        epochs=history.epoch,
        metrics_history= pd.DataFrame(history.history),
    )

print("Defindd the create_model and train_model")