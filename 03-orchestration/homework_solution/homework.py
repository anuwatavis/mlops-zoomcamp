import pickle
import pandas as pd
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.task_runners import SequentialTaskRunner
import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import os


@task()
def read_data(path):
    logger = get_run_logger()
    logger.info(f"Read data from {path}")
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):

    print("Run model")
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


def download_fhv_file(file_name: str):
    URL_TO_DOWNLOAD = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{file_name}.parquet'
    os.system(f"wget {URL_TO_DOWNLOAD} -P ./data")


@task
def get_path(date=None):

    date_ = datetime.date.today()

    if date is not None:
        date_ = datetime.datetime.strptime(date, '%Y-%m-%d')

    training_date = (date_ - relativedelta(months=2)
                     ).strftime("%Y-%m")
    valid_date = (date_ - relativedelta(months=1)).strftime("%Y-%m")

    training_path = f"./data/fhv_tripdata_{training_date}.parquet"
    valid_path = f"./data/fhv_tripdata_{valid_date}.parquet"

    # Check traning and valid path exist
    training_file_is_exist = Path(training_path).exists()
    valid_file_is_exist = Path(valid_path).exists()

    if not training_file_is_exist:
        download_fhv_file(training_date)

    if not valid_file_is_exist:
        download_fhv_file(valid_date)

    return training_path, valid_path


@flow(task_runner=SequentialTaskRunner())
def main(date: str = None):
    train_path, val_path = get_path(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # save model and dictvectorizer
    with open(f"./artifacts/models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f"./artifacts/vectorizers/dv-{date}.bin", "wb") as f_out:
        pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main,
    name="cron-schedule-deployment_2",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
)
