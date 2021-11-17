import mlflow.sklearn
from fbprophet import Prophet
import numpy as np
import pandas as pd
import shutil
import os

from datetime import datetime, timedelta
# mlflow.sklearn.autolog()

def noise(size: int):
    return np.random.rand(size)

def toDateFrom(origin: datetime, days: int):
    return (origin + timedelta(days=days)).strftime('%Y-%m-%d')

def prepare_data():
    size = 366 * 5
    days = np.arange(0, size, 1)
    quarter = np.sin(np.pi * days / 90.) * 2.5 + noise(size) * 0.02
    week = np.sin(np.pi * days / 7) * 1 + noise(size) * 0.05
    month = np.sin(np.pi * days / 30) * 2 + noise(size) * 0.2
    year = np.sin(np.pi * days / 366) * 4 + noise(size) * 0.3
    y = quarter * 0.2 + np.cos(week) + np.exp(month) + year * 2.2
    concat = np.concatenate([[days, quarter, week, month, year, y]], axis=1)
    df = pd.DataFrame(data=concat.T, columns=['ds', 'quarter', 'week', 'month', 'year', 'y'])
    df['ds'] = df['ds'].apply(lambda x: toDateFrom(datetime(year=2015, month=1, day=1), x))
    print(df)
    return df

def udf(interval_width: float):
    model = Prophet(
        interval_width=interval_width,
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True)
    data = prepare_data()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)
    result = model.predict(future)
    print(result)
    filepath = f'./prophet.model-{interval_width}'
    if os.path.isdir(filepath):
        shutil.rmtree(filepath)
    mlflow.sklearn.save_model(model, filepath)


if __name__ == '__main__':
    udf(0.82)
