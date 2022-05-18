

import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt 
import warnings
warnings.filterwarnings("ignore")



def getting_data():
    df_january = pd.read_parquet (r'D:\Python\Projects\NL_Automated_Reports\Testing\Data\fhv_tripdata_2021-01.parquet')
    df_february = pd.read_parquet (r'D:\Python\Projects\NL_Automated_Reports\Testing\Data\fhv_tripdata_2021-02.parquet')
    df = pd.concat([df_january, df_february])
    return df

def new_features (df):
    df['duration']= ((df.dropOff_datetime - df.pickup_datetime ).dt.total_seconds())/60
    return df 

def dtypes_transformation(df):
    df.PUlocationID = df.PUlocationID.astype(str)
    df.DOlocationID = df.DOlocationID.astype(str)
    return df
    

def feature_transformation(X,cat_features):
    # Define Transformers                                                                       
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=True)
    
    # Define steps of the pipeline
    t = [(('cat', categorical_transformer, cat_features))]
    preprocesor = ColumnTransformer(transformers = t,remainder='passthrough' )
    steps = [('preprocesor',preprocesor),]
    
    # define the pipeline
    pipeline = Pipeline(steps=steps) 
    X_scaled = pipeline.fit_transform(X)
    return X_scaled

def pre_processing(df):
    df = dtypes_transformation(df)
    df= df[(df.duration>=1) & (df.duration<=60)]
    X = df[['PUlocationID', 'DOlocationID', 'duration']]
    X_valid,y = X.drop('duration', axis=1), X.duration
    cat_features = ['PUlocationID', 'DOlocationID']
    X_scaled = feature_transformation(X_valid ,cat_features)
    return X_scaled,y



def main():
    
    df = getting_data()
    df = new_features(df)
    
    df_january = df[df.pickup_datetime.dt.month==1]
    df_february = df[df.pickup_datetime.dt.month==2]
    
    X_train_scaled,y_train = pre_processing(df_january)
    X_valid_scaled, y_valid = pre_processing(df_february)
    
        
    # Question 1: Number of records in Jan 2021 FHV data
    print ("Number of records in Jan 2021 FHV data is {} rows".format(df_january.shape[0]))
    
    # Question 2: Average duration in Jan 2021 FHV
    print("Average duration in Jan 2021 FHV is {:.2f} min".format(df_january.duration.mean()))
    
    # Question 3: Fraction of missing values
    print("Fraction of missing values is {}".format(df_january.isnull().sum()*100/df_january.shape[0]))
    
    # Question 4: Dimensionality after OHE
    print("Dimensionality after OHE is {}".format(X_valid_scaled.shape[1]))
    
    # Question 4 : RMSE of the trained model
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
       

