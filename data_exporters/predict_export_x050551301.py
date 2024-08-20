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
import tempfile


import warnings
from default_repo.utils.feature_extraction.skPCA import skpca
from default_repo.utils.feature_extraction.skTSNE import sktsne
# from default_repo.feature_extraction.sUMAP import sumap
from default_repo.utils.feature_extraction.skLDA import sklda
from default_repo.utils.feature_extraction.skRP import skrp
from default_repo.utils.feature_extraction.skFH import skfh
from default_repo.utils.feature_extraction.skIncPCA import skincpca

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


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



def load_config():
    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        return config
    
    return None

@data_exporter
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
        run_id="c46597e59a4c425aa9d363bf91bb1e0b"
        # run_id="273d779069184b758690c4fa143a8493" #ludwig
        

    
    # logged_model=kwargs.get('logged_model')
    model_type=kwargs.get('model_type')

    if model_type is None:
        logged_model = f'runs:/{run_id}/water_model'   #lgbm
        # logged_model=f'runs:/{run_id}/my_ludwig_experiment_model' #ludwig
    elif model_type=='Ludwig':
        logged_model=f'runs:/{run_id}/my_ludwig_experiment_model' #ludwig

    elif model_type=='LGBM':
        logged_model = f'runs:/{run_id}/water_flow_model'   #lgbm

    # logged_model = f'runs:/{run_id}/water_flow_model'   #lgbm
    # logged_model=f'runs:/{run_id}/my_ludwig_experiment_model' #ludwig


    print(logged_model)

    loaded_model = mlflow.pyfunc.load_model(logged_model)

    model = mlflow.pyfunc.load_model(logged_model)

    if 'my_ludwig_experiment_model' in logged_model:
        # add observedAt input for ludwig model
        input_df['observedAt']=input_df.index

    print(input_df.columns)


    features=input_df.columns

    
    pca = skpca(n_components=2) 

    
    pca_result = pca.fit_transform(input_df[features])
    print(f"pca result {pca_result}")
    
    df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    df_pca.index=input_df.index


    y_pred = model.predict(df_pca)

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

    # plt.figure(figsize=(10, 6))
    plt.plot(df_compare['date'].tail(20), df_compare['X050551301_predicted_flow'].tail(20), linestyle='-')


    # plt.gca().xaxis.set_major_locator(DayLocator(interval=300))  
    plt.gcf().autofmt_xdate()

    plt.xlabel('Date')
    plt.ylabel('X050551301 Predicted Flow')
    plt.title('Plot of Dates and Predicted Water Flow')
    plt.xticks(rotation=45)
    # plt.gcf().set_size_inches(12, 8)

    # plt.show()


    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "prediction_plot_x050551301.png")
        plt.savefig(file_path)


        # save to minio
        config = load_config()
        client = Minio(
            config["default"]["MINIO_HOST"],
            access_key=config["default"]["MINIO_ACCESS_KEY"],
            secret_key=config["default"]["MINIO_SECRET_KEY"],
        )

        if not client.bucket_exists("lgboost-model"):
            client.make_bucket("lgboost-model")
            
        client.fput_object("lgboost-model", "prediction_plot_x050551301.png", file_path)

    return df_compare
