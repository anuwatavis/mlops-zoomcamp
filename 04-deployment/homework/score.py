#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd
import sys
import os


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def run(month: str, year: str):
    # Read data
    df = read_data(
        f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    mean_predicted_duration = y_pred.mean()

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df['predictions'] = y_pred

    df_result = df[['ride_id', 'predictions']]

    df_result.to_parquet(
        "./result.parquet",
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(f'(score): mean predited duration=[{mean_predicted_duration}]')


if __name__ == '__main__':
    year = int(os.environ.get('YEAR'))
    month = int(os.environ.get("MONTH"))
    run(month=month, year=year)
