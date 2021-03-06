
#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import pickle
import os
import mlflow
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
#%%

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

#%%
def main():
    mlflow.set_tracking_uri(r"file\\\D:\Python\Projects\NL_Automated_Reports\Testing\mlflow")
    run_name = mlflow.set_experiment("mlflow_experiment2")
    dir_name = Path(__file__).parents[1]
    with mlflow.start_run(run_name=run_name,nested=True):
        mlflow.set_tags({"model": "linear", "encode": "one_hot"})
        mlflow.log_param("model", "linear")
        mlflow.log_artifact(os.path.join(dir_name, 'Model'))
        df = getting_data(dir_name)
        df = new_features(df)

        df_january = df[df.pickup_datetime.dt.month == 1]
        df_february = df[df.pickup_datetime.dt.month == 2]

        cat_features = ['PUlocationID', 'DOlocationID']

        # Question 4: Dimensionality after OHE
        df_january = pre_processing(df_january)
        X_train, y_train = get_x_y(df_january)
        X_train_scaled = feature_transformation(X_train, cat_features, dir_name)
        
        # Question 5 : RMSE of the trained model
        regressor = LinearRegression()
        model = regressor.fit(X_train_scaled, y_train)
        predictions = model.predict(X_train_scaled)
        
        rmse = sqrt(mean_squared_error(y_train, predictions))
        mlflow.log_metric("RMSE", rmse)


    """ df_february = pre_processing(df_february)
    X_valid, y_valid = get_x_y(df_february)
    preprocessing = joblib.load(os.path.join(
        dir_name, 'Model', 'preprocesor.joblib'))
    X_valid_scaled = preprocessing.transform(X_valid) """

    


if __name__ == '__main__':
    main()

# %%
