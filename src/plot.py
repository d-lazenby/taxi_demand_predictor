from typing import Optional
from datetime import timedelta

import plotly.express as px
import pandas as pd

def plot_one_sample(
    features: pd.DataFrame,
    targets: pd.Series,
    example_id: int, 
    predictions: Optional[pd.Series] = None,
):
    features_ = features.iloc[example_id]
    target_ = targets.iloc[example_id]
    
    ts_columns = [col for col in features.columns if col.startswith('rides_previous')]
    ts_values = [features_[col] for col in ts_columns] + [target_]
    
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='h'
    )
    
    # line plot with past values
    title = f"Pickup hour={features_['pickup_hour']}, location_id={features_['pickup_location_id']}"
    fig = px.line(
        x=ts_dates, 
        y=ts_values,
        template='plotly_dark',
        markers=True, 
        title=title
    )
    
    # Green dot for the value we want to predict
    fig.add_scatter(
        x=ts_dates[-1:],
        y=[target_],
        line_color='green',
        mode='markers',
        marker_size=10,
        name='actual_value'
    )
    
    if predictions is not None:
        # Big red cross for predicted value
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=ts_dates[-1:],
            y=[prediction_],
            line_color='red',
            mode='markers',
            marker_symbol='x',
            marker_size=15,
            name='prediction'
        )
    
    return fig