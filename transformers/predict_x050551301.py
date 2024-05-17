# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mage_ai.data_preparation.variable_manager import get_variable
import os
import yaml
import mlflow
from mage_ai.settings.repo import get_repo_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.font_manager
# print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
from matplotlib.dates import DateFormatter,DayLocator
from minio import Minio


import warnings

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

warnings.filterwarnings("ignore", category=UserWarning)

# plt.rcParams['font.family']='Arial'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def start_mlflow():
    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['MLFLOW_TRACKING_USERNAME'] = config['default']['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config['default']['MLFLOW_TRACKING_PASSWORD']
    os.environ['AWS_ACCESS_KEY_ID'] = config['default']['AWS_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = config['default']['AWS_SECRET_ACCESS_KEY']
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = config['default']['MLFLOW_S3_ENDPOINT_URL']
    os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = config['default']['MLFLOW_TRACKING_INSECURE_TLS']
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"

    mlflow.set_tracking_uri("http://62.72.21.79:5000")



@transformer
def transform_custom(input_df, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    print(input_df)
    print(input_df.columns)
    # input_df['observedAt']=input_df.index

    # print(mlflow.get_experiment_by_name("water_flow"))
    print(mlflow.get_experiment_by_name("water_flow_lgbm"))

    # experiment_name = 'water_flow_model'




    run_id=kwargs.get('run_id')

    if run_id is None:
        run_id = "2998cc1d88e34de1a9d683f40ac5b504"  #lgbm
        # run_id="273d779069184b758690c4fa143a8493" #ludwig
        

    
    # logged_model=kwargs.get('logged_model')
    model_type=kwargs.get('model_type')

    if model_type is None:
        logged_model = f'runs:/{run_id}/water_flow_model'   #lgbm
        # logged_model=f'runs:/{run_id}/my_ludwig_experiment_model' #ludwig
    elif model_type=='Ludwig':
        logged_model=f'runs:/{run_id}/my_ludwig_experiment_model' #ludwig

    elif model_type=='LGBM':
        logged_model = f'runs:/{run_id}/water_flow_model'   #lgbm

    # logged_model = f'runs:/{run_id}/water_flow_model'   #lgbm
    # logged_model=f'runs:/{run_id}/my_ludwig_experiment_model' #ludwig


    print(logged_model)


    model = mlflow.pyfunc.load_model(logged_model)

    if 'my_ludwig_experiment_model' in logged_model:
        # add observedAt input for ludwig model
        input_df['observedAt']=input_df.index


    y_pred = model.predict(input_df)

    # print(y_pred)


    # y_pred, _ = model.predict(input_df)

    target_column='X050551301'


    # y_pred.reset_index(drop=True,inplace=True)
    df_compare=pd.DataFrame()
    df_compare['date']=input_df.index

    try:
        df_compare['X050551301_predicted_flow']=y_pred
    except ValueError:
        # y_pred.columns = ['X050551301_predictions']
        df_compare['X050551301_predicted_flow']=y_pred[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df_compare['date'], df_compare['X050551301_predicted_flow'], linestyle='-')


    plt.gca().xaxis.set_major_locator(DayLocator(interval=50))  # adjust space between dates
    plt.gcf().autofmt_xdate()

    plt.xlabel('Date')
    plt.ylabel('X050551301 Predicted Flow')
    plt.title('Plot of Dates and Predicted Water Flow')
    plt.xticks(rotation=45)

    plt.show()

    return df_compare
