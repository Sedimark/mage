from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import logging

import os
import time
import lightgbm
import re
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from os import path
import yaml
from sklearn.model_selection import train_test_split
import mlflow
import warnings
from mage_ai.settings.repo import get_repo_path
import numpy as np

from sklearn.model_selection import GridSearchCV
from mage_ai.data_preparation.variable_manager import get_variable
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler

# pca
from default_repo.utils.feature_extraction.skPCA import skpca
from default_repo.utils.feature_extraction.skTSNE import sktsne
# from default_repo.feature_extraction.sUMAP import sumap
from default_repo.utils.feature_extraction.skLDA import sklda
from default_repo.utils.feature_extraction.skRP import skrp
from default_repo.utils.feature_extraction.skFH import skfh
from default_repo.utils.feature_extraction.skIncPCA import skincpca

# warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

# import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

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
  
#   return plt

def plot_real_pred(y_test,y_pred):
    plt.figure()
    plt.scatter(y_test,y_pred, c='green')

    plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],linestyle='--',color='crimson')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)


    plt.legend()
    # plt.show()
    figure_path = "water_flow_model_vs.png"
    plt.savefig(figure_path)

def mae_score(y_true,y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def calculate_precision(y_true, y_pred):
    result = []
    for i in range(len(y_true)):
        if y_true[i] != 0:
            abs_difference = abs(y_true[i] - y_pred[i])
            result.append(abs_difference / y_true[i])
    if len(result) == 0:
        return np.nan  
    mean_precision = np.mean(result)
    return mean_precision
    



module_dict = {
 # pca libraries
'skpca': skpca,
'sktsne':sktsne,
# 'sumap':sumap,
'sklda':sklda,
'skrp':skrp,
'skfh':skfh,
'skincpca':skincpca
}


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


    # data_precipitation = get_variable('flawless_waterfall', 'load_precipitation', 'output_0')
    # print(data[1])

    pca_module_name=data[1]

    print(pca_module_name)


    # pca
    # pca_module = module_dict.get(pca_module_name)  
    # pca = pca_module(n_components=2) 
    # pca

    df_new=data[0].copy()

    # df_new = pd.concat([df_new,data_precipitation], axis=1)
 
    # make nan values 0

    df_new[df_new['X050551301'].isnull()]=0


    target_column = 'X050551301'

    # exclude target from input
    features = df_new.drop(columns=[target_column]).columns

    scaler = StandardScaler()
    df_new[features] = scaler.fit_transform(df_new[features])


    print(f'input features: {features}')


    print(f'target column: {target_column}')

    # pca
    pca_module = module_dict.get(pca_module_name)  
    pca = pca_module(n_components=2) 

    # pca = skpca(n_components=2)

    # 
    df_new[target_column] = pd.to_numeric(df_new[target_column], errors='coerce')
    # 

    
    pca_result = pca.fit_transform(df_new[features])
    print(f"pca result {pca_result}")
    
    df_pca = pd.DataFrame(data=np.column_stack((pca_result, df_new[target_column])), columns=['PCA1', 'PCA2', target_column])



    # 

    # check if this fixes the issue
    df_pca['PCA1'] = pd.to_numeric(df_pca['PCA1'], errors='coerce')
    df_pca['PCA2'] = pd.to_numeric(df_pca['PCA2'], errors='coerce')

    # 


    df_pca.index=df_new.index

    df_new=df_pca.copy()


    features = df_pca.drop(columns=[target_column]).columns



    # pca



    param_grid = {
    'learning_rate': [0.1, 0.3, 0.5,0.03],
    'n_estimators': [50, 100, 200],
        'max_depth': [-1,3, 5],
        'num_leaves': range(20, 101, 20),  
        # 'min_data_in_leaf': range(10, 51, 10),
        # 'feature_fraction': [0.6, 0.8, 1.0],
        # 'reg_alpha': [0.0, 0.1, 1.0]  # regularization
    }



    model = GridSearchCV(LGBMRegressor(random_state=None), param_grid=param_grid,  scoring=mae_score)

    train_data, test_data = train_test_split(df_new, test_size=0.2, shuffle=False)


    X_train, y_train = train_data[features], train_data[target_column]
    X_test, y_test = test_data[features], test_data[target_column]


    model.fit(X_train, y_train)

    
    best_model = model.best_estimator_
    best_params = model.best_params_


    y_pred = best_model.predict(X_test)


    mean_precision=calculate_precision(y_test,y_pred)




    mae = np.round(mean_absolute_error(y_test, y_pred), 3)



    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error


    r_squared = r2_score(y_test, y_pred)



    scaler = StandardScaler()
    y_test_scaled = scaler.fit_transform(np.array(y_test).reshape(-1, 1)).flatten()
    y_pred_scaled = scaler.transform(np.array(y_pred).reshape(-1, 1)).flatten()

    # Calculate MSE on scaled values
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse_scaled=np.sqrt(mse_scaled)

    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


    
    metrics={
            "mse":mse,
            "mse_scaled":mse_scaled,
            "rmse":rmse,
            "rmse_scaled":rmse_scaled,

            "mae":mae,
            "r_squared":r_squared,
            "mape":mape,
            "mean_precision":mean_precision
            }


    y_test.reset_index(drop=True,inplace=True)


    df_compare=pd.DataFrame()
    df_compare['y_pred']=y_pred
    df_compare['y_test']=y_test
    print(f"df_compare {df_compare}")
    
    # Infer the model signature-MLFLOW
    signature = infer_signature(X_test, y_pred)

    plot_predictions(test_data,y_test,y_pred)
    plot_real_pred(y_test,y_pred)

    client = MlflowClient()

    # client.create_registered_model("water_model",tags={"model_type": "LGBM", "mage_model": "true"})

    model_name = "water_model"
    try:
        registered_model = client.get_registered_model(model_name)
    except Exception as e:
        registered_model = client.create_registered_model(model_name, tags={"model_type": "LGBM", "mage_model": "true"})


    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("water").experiment_id) as run:
        mlflow.sklearn.log_model(
            sk_model=best_model,
            # artifact_path="water_flow_model",
            artifact_path="water_model",

            registered_model_name="water_model", 
            signature=signature,

        )

        mlflow.log_artifact("water_flow_model.png", artifact_path="figures")
        mlflow.log_artifact("water_flow_model_vs.png", artifact_path="figures")

        for k, v in metrics.items():
            mlflow.log_metrics({k: v})

        mlflow.set_tag("model_type", "LightGBM")
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


    return df_compare







@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'