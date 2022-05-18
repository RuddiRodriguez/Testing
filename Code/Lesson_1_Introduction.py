

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def getting_data(dir_name):
    df_january = pd.read_parquet(os.path.join(
        dir_name, 'Data', 'fhv_tripdata_2021-01.parquet'))
    df_february = pd.read_parquet(os.path.join(
        dir_name, 'Data', 'fhv_tripdata_2021-02.parquet'))
    df = pd.concat([df_january, df_february])
    return df


def get_x_y(df):
    X, y = df.drop('duration', axis=1), df['duration']
    return X, y


def new_features(df):
    df['duration'] = (
        (df.dropOff_datetime - df.pickup_datetime).dt.total_seconds())/60
    return df


def dtypes_transformation(df):
    df.PUlocationID = df.PUlocationID.astype(str)
    df.DOlocationID = df.DOlocationID.astype(str)
    return df


def feature_transformation(X, cat_features, dir_name):

    # Define Transformers
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore', sparse=True)

    # Define steps of the pipeline
    t = [(('cat', categorical_transformer, cat_features))]
    preprocesor = ColumnTransformer(transformers=t, remainder='passthrough')
    steps = [('preprocesor', preprocesor), ]

    # define the pipeline
    pipeline = Pipeline(steps=steps)
    X_scaled = pipeline.fit_transform(X)
    joblib.dump(preprocesor, os.path.join(
        dir_name, 'Model', 'preprocesor.joblib'))
    return X_scaled


def pre_processing(df):
    df[['PUlocationID', 'DOlocationID']] = df[[
        'PUlocationID', 'DOlocationID']].fillna(value=-1)
    df = dtypes_transformation(df)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    return df[['PUlocationID', 'DOlocationID', 'duration']]


def main():
    dir_name = Path(__file__).parents[1]

    df = getting_data(dir_name)
    df = new_features(df)

    df_january = df[df.pickup_datetime.dt.month == 1]
    df_february = df[df.pickup_datetime.dt.month == 2]

    cat_features = ['PUlocationID', 'DOlocationID']

    # Question 1: Number of records in Jan 2021 FHV data
    print("Number of records in Jan 2021 FHV data is {} rows".format(
        df_january.shape[0]))

    # Question 2: Average duration in Jan 2021 FHV
    print("Average duration in Jan 2021 FHV is {:.2f} min".format(
        df_january.duration.mean()))

    # Question 3: Fraction of missing values
    print("Fraction of missing values is {:.2f}".format(
        df_january['PUlocationID'].isnull().sum()*100/df_january['PUlocationID'].shape[0]))

    # Question 4: Dimensionality after OHE
    df_january = pre_processing(df_january)
    X_train, y_train = get_x_y(df_january)
    X_train_scaled = feature_transformation(X_train, cat_features,dir_name)

    df_february = pre_processing(df_february)
    X_valid, y_valid = get_x_y(df_february)
    preprocessing = joblib.load(os.path.join(
        dir_name, 'Model', 'preprocesor.joblib'))
    X_valid_scaled = preprocessing.transform(X_valid)

    print("Dimensionality of training data after OHE is {}".format(
        X_train_scaled.shape[1]))
    print("Dimensionality of validation data after OHE is {}".format(
        X_valid_scaled.shape[1]))

    # Question 5 : RMSE of the trained model
    regressor = LinearRegression()
    model = regressor.fit(X_train_scaled, y_train)
    predictions = model.predict(X_train_scaled)
    rmse = sqrt(mean_squared_error(y_train, predictions))
    print("RMSE of the trained model is {:.2f}".format(rmse))

    # Question 6: RMSE on validation
    predictions = model.predict(X_valid_scaled)
    rmse = sqrt(mean_squared_error(y_valid, predictions))
    print("RMSE on validation is {:.2f}".format(rmse))


if __name__ == '__main__':
    main()
