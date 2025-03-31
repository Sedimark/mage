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

from default_repo.utils.feature_extraction.skPCA import skpca
from default_repo.utils.feature_extraction.skTSNE import sktsne
from default_repo.utils.feature_extraction.sUMAP import sumap
from default_repo.utils.feature_extraction.skLDA import sklda
from default_repo.utils.feature_extraction.skRP import skrp
from default_repo.utils.feature_extraction.skFH import skfh
from default_repo.utils.feature_extraction.skIncPCA import skincpca

# pca
from default_repo.utils.feature_extraction.skPCA import skpca
from default_repo.utils.feature_extraction.skTSNE import sktsne
from default_repo.utils.feature_extraction.sUMAP import sumap
from default_repo.utils.feature_extraction.skLDA import sklda
from default_repo.utils.feature_extraction.skRP import skrp
from default_repo.utils.feature_extraction.skFH import skfh
from default_repo.utils.feature_extraction.skIncPCA import skincpca

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def plot_test_data(df_compare):
    df_compare.plot(kind='scatter', x='y_pred', y='y_test', s=32, alpha=.8)
    plt.gca().spines[['top', 'right',]].set_visible(False)

    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Comparison between Predicted and Actual Values')



def simulate_future_data(df, periods=30):
    """
    Simulates future data for the next 'periods' days based on the average change observed
    in the most recent data points of the dataset.

    Args:
    df (pd.DataFrame): The original dataset.
    periods (int): Number of future periods to simulate data for.

    Returns:
    pd.DataFrame: A DataFrame containing simulated future data.
    """

    df.index = pd.to_datetime(df.index)


    future_dates = pd.date_range(start=df.index.max(), periods=periods + 1)[1:]
    future_df = pd.DataFrame(index=future_dates)

    change_columns = df.columns#['X031001001', 'X050551301', 'X051591001','rain (mm)']
    recent_df = df[change_columns].tail(7)
    daily_change = recent_df.diff().mean()

    for col in change_columns:
        future_df[col] = df[col].iloc[-1] + daily_change[col] * np.arange(1, periods + 1)

    return future_df#.reset_index().rename(columns={'index': 'observedAt'})



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



module_dict = {
 # pca libraries
'skpca': skpca,
'sktsne':sktsne,
'sumap':sumap,
'sklda':sklda,
'skrp':skrp,
'skfh':skfh,
'skincpca':skincpca
}

@custom
def transform_custom(compare_df, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    print(compare_df)
    # data = get_variable('flawless_waterfall', 'impute_missing_water_flow', 'output_0')

    data = get_variable('ml_flow_lightgbm', 'dimensionality_reduction', 'output_0')



    future_data = simulate_future_data(data,periods=20)



    start_mlflow()

    print(mlflow.get_experiment_by_name("water_flow"))


    # run_id="2346034608bf4846bcb044f16c95cd9c"
    run_id="1f6b0af40f8646c09262eb80e24eccf3"

    logged_model = f'runs:/{run_id}/water_model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)


    print(logged_model)


    model = mlflow.pyfunc.load_model(logged_model)

    print(future_data.columns)

# ***********************
    target_column = 'X050551301'

    # exclude target from input
    features = future_data.drop(columns=[target_column]).columns


    # test
    pca_module_name = get_variable('ml_flow_lightgbm', 'dimensionality_reduction', 'output_1')

    pca_module = module_dict.get(pca_module_name)  
    pca = pca_module(n_components=2) 


    # 

    # pca = skpca(n_components=2) 

    
    pca_result = pca.fit_transform(future_data[features])
    print(f"pca result {pca_result}")
    
    df_pca = pd.DataFrame(data=np.column_stack((pca_result, future_data[target_column])), columns=['PCA1', 'PCA2', target_column])
    df_pca.index=future_data.index
# ********************

    y_pred = model.predict(df_pca)
    # y_pred = model.predict(future_data)




    df_compare=pd.DataFrame()
    df_compare['date']=future_data.index
    df_compare['X050551301_predicted_flow']=y_pred
    return df_compare





@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
