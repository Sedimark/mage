from datetime import datetime
import pandas as pd
from ludwig.utils.data_utils import add_sequence_feature_column
import logging
from ludwig.api import LudwigModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from os import path
import yaml
from sklearn.model_selection import train_test_split
import numpy as np
from mage_ai.settings.repo import get_repo_path

import mlflow
import os
import time
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test




def mae_score(y_true,y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def calculate_precision(y_true, y_pred):
    result = []
    for i in range(len(y_true)):
        if i < len(y_pred) and y_true[i] != 0:
            if i < len(y_pred.index):
                abs_difference = abs(y_true[i] - y_pred.iloc[i][0])  
                result.append(abs_difference / y_true[i])
            else:
                result.append(np.nan)
    if len(result) == 0:
        return np.nan  
    mean_precision = np.nanmean(result)  
    return mean_precision


config_path = path.join(get_repo_path(), 'io_config.yaml')
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


MLFLOW_TRACKING_USERNAME = config['default']['MLFLOW_TRACKING_USERNAME']
MLFLOW_TRACKING_PASSWORD = config['default']['MLFLOW_TRACKING_PASSWORD']
AWS_ACCESS_KEY_ID = config['default']['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = config['default']['AWS_SECRET_ACCESS_KEY']
MLFLOW_S3_ENDPOINT_URL = config['default']['MLFLOW_S3_ENDPOINT_URL']
MLFLOW_TRACKING_INSECURE_TLS = config['default']['MLFLOW_TRACKING_INSECURE_TLS']


os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = MLFLOW_TRACKING_INSECURE_TLS
os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"

mlflow.set_tracking_uri("http://62.72.21.79:5000")


def plot_predictions(test_data,y_test,y_pred):
  plt.figure()
  plt.plot(test_data.index,y_test, label='Actual Water Flow')
  plt.plot(test_data.index,y_pred, label='Predicted Water Flow', linestyle='--')
  plt.ylabel('Test Water Flow')
  plt.legend()
  figure_path = "water_flow_model.png"
  plt.savefig(figure_path)

def plot_real_pred(y_test,y_pred):
    plt.figure()
    plt.scatter(y_test,y_pred, c='green')
    plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],linestyle='--',color='crimson')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.legend()

    figure_path = "water_flow_model_vs.png"
    plt.savefig(figure_path)

@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    start_time = time.time()

    df_new=data.copy()


    df_new['observedAt']=df_new.index
    df_new.reset_index(inplace=True,drop=True)

    target_column = 'X050551301'

    # exclude target from input
    features = df_new.drop(columns=[target_column]).columns


    config = {
    "input_features": [
        {
        "name": "X031001001",
        "type": "timeseries",
        },
              {
        "name": "X045401001",
        "type": "timeseries",
        },
              {
        "name": "X051591001",
        "type": "timeseries",
        },
          {
        "name": "observedAt",
        "type": "date",
        }

    ],
    "output_features": [
        {
        "name": "X050551301",
        "type": "number",
        }
    ],

        
    "preprocessing": { 
    "scaler": "standard"
    },
        
    "model": {
    "type": "stacked_lstm", 
    "num_layers": 2, 
    "num_units": 64,  
    "bidirectional": True, 
    "dropout": 0.3  

    },
        
    "trainer": {
    "epochs": 100,
    # "learning_rate": 0.03,
    "validation_field": "X050551301",
      "validation_metric": "r2",
    },     
    }


    model = LudwigModel(config, logging_level=logging.INFO)

    train_data, test_data = train_test_split(df_new, test_size=0.1, shuffle=False)

    X_train, y_train = train_data[features], train_data[target_column]
    X_test, y_test = test_data[features], test_data[target_column]

    
    train_size = int(0.8 * len(df_new))
    vali_size = int(0.1 * len(df_new))

    
    # train, validation, test split
    df_new['split'] = 0
    df_new.loc[
        (
            (df_new.index.values >= train_size) &
            (df_new.index.values < train_size + vali_size)
        ),
        ('split')
    ] = 1
    df_new.loc[
        df_new.index.values >= train_size + vali_size,
        ('split')
    ] = 2


    train_stats, preprocessed_data, output_directory = model.train(
        # training_set=train_data, test_set=test_data,  
        dataset=df_new,
        output_directory='results',

        )


    y_pred, _ = model.predict(dataset=test_data)

    target_column='X050551301'

    df_comparison=pd.DataFrame()


    y_pred.reset_index(drop=True,inplace=True)
    y_test.reset_index(inplace=True,drop=True)


    df_comparison[f'{target_column}_recorded']=test_data[target_column]


    y_pred.columns = ['X050551301_predictions']


    df_comparison[f'{target_column}_predicted']=y_pred['X050551301_predictions'].values
    mean_precision=calculate_precision(y_test,y_pred)


    mae = np.round(mean_absolute_error(y_test, y_pred), 3)
    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)  

    r_squared = r2_score(y_test, y_pred)


    
    scaler = StandardScaler()
    y_test_scaled = scaler.fit_transform(np.array(y_test).reshape(-1, 1)).flatten()
    y_pred_scaled = scaler.transform(np.array(y_pred).reshape(-1, 1)).flatten()

    # Calculate MSE on scaled values
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse_scaled=np.sqrt(mse_scaled)



    metrics={  
            "mse":mse,
            "mse_scaled":mse_scaled,
            "rmse":rmse,
            "rmse_scaled":rmse_scaled,
            "mae":mae,
            "r_squared":r_squared,
            "mean_precision":mean_precision,

    }

    print(f'metrics :{metrics}')


    signature = infer_signature(X_test, y_pred)

    plot_predictions(test_data,y_test,y_pred)
    plot_real_pred(y_test,y_pred)

    client = MlflowClient()

    model_name = "water_model"
    try:
        registered_model = client.get_registered_model(model_name)
    except Exception as e:
        registered_model = client.create_registered_model(model_name, tags={"model_type": "Ludwig AI", "mage_model": "true"})


    # mlflow
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("water").experiment_id) as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="my_ludwig_experiment_model",
            registered_model_name="water_flow_model_ludwig",
            signature=signature,
        )

        mlflow.log_artifact("water_flow_model.png", artifact_path="figures")
        mlflow.log_artifact("water_flow_model_vs.png", artifact_path="figures")

        for k, v in metrics.items():
            mlflow.log_metrics({k: v})


        mlflow.set_tag("model_type", "Ludwig AI")
        mlflow.set_tag("dataset", "historic waterFlow from 1960-2023")

        EXPERIMENT_DESCRIPTION='This model uses as input water flow data from stations:X031001001,X045401001,X05159100 and predicts the water flow for X050551301'
        mlflow.set_tag('mlflow.note.content',EXPERIMENT_DESCRIPTION)
        run_id = run.info.run_id


    src_uri=f"runs://{run_id}/water_model"
    result = client.create_model_version(
        name=model_name,
        source=src_uri,
        run_id=run_id,
    )
        
    end_time = time.time()
    
    elapsed_time = end_time - start_time

    print(f"elapsed time: {elapsed_time}")
        
        
    return df_comparison#,train_stats


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'