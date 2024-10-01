import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline

import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column indicating the average demand observed at t - 7 days, t - 14 days, t - 21 days and t - 28 days
    e.g. if we want to predict Friday at 6pm, we look at Friday at 6pm for the previous four weeks and take the average.
    """
    X['average_rides_last_4_weeks'] = 0.25 * (
            X[f'rides_previous_{7*24*1}_hour'] + \
            X[f'rides_previous_{7*24*2}_hour'] + \
            X[f'rides_previous_{7*24*3}_hour'] + \
            X[f'rides_previous_{7*24*4}_hour']
        )
    
    return X

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        X_['pickup_hour'] = pd.to_datetime(X_['pickup_hour'])
        
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        
        return X_.drop(columns=['pickup_hour'])
    

def get_pipeline(**hyperparams: dict) -> Pipeline:

    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False
    )
    
    add_temporal_features = TemporalFeatureEngineer()
    
    return make_pipeline(
        add_feature_average_rides_last_4_weeks, 
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )