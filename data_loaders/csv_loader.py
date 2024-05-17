import pandas as pd
import requests
import io
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test



def clean_data(file_contents):
    """
    Clean csv dataset before passing it in the pipeline
    """
    df = pd.read_csv(file_contents)

    # Extract the measurement unit from the second column
    df["measurement"] = df[df.columns[1]].str.split().str[0]
    df["measurementUnit"] = df[df.columns[1]].str.split().str[1]

    # drop second column, after having created 2 new ones
    df = df.drop(df.columns[[1]], axis=1)  # drop second column -> illuminance/temperature/etc.

    # remove % in humidity dataset
    df[df.columns[1]] = df[df.columns[1]].str.replace('%', '').astype(float)

    df = df.dropna(axis=1)

    # print('Clean df is \n', df)
    print('type ', type(df))

    return df#, missing_info


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    csv_filepath='default_repo/data_loaders/input/Temperature-data-23_02_2023 14 54 17.csv'


    # clean data API
    server_url = "http://host.docker.internal:8080" 
    # server_url = "http://localhost:8080" 


    csv_filepath_mean='default_repo/data_loaders/input/dataset_mean.csv'


    with open(csv_filepath_mean, "rb") as file:
        file_content = file.read()

    # make mean of data
    payload = {"file": ("dataset_mean.csv", file_content, "multipart/form-data")}

    # response = requests.post(f"{server_url}/mean", files=payload)

    
    # Check the response of the means endpoint
    # if response.status_code == 200:
    #     data = response.json()
    #     print("Message:", data["message"])
    #     print("Means:")
    #     for column, mean in data["means"].items():
    #         print(f"{column}: {mean}")
    # else:
    #     print("Error:", response.status_code)
    #     print(response.text)



    # with open(csv_filepath, "rb") as csv_file:
    #     files = {'file': ('data.csv', csv_file)}

    #     response = requests.post(f"{server_url}/clean", files=files)

    #     if response.status_code == 200:
    #         data = response.json()
    #         print("Message:", data["message"])
    #         cleaned_df_req = data.get("cleaned_df") 
    #         if cleaned_df_req:
    #             cleaned_df = pd.read_json(cleaned_df_req)
    #             return cleaned_df
    #         else:
    #             print("No cleaned_df data found in the response.")
    #     else:
    #         print("Request to /clean failed with status code:", response.status_code)
    #         print("Response content:", response.content)


    # with open(csv_filepath, "rb") as csv_file:
    #     files = {'file': ('data.csv', csv_file)}

    #     response = requests.post(f"{server_url}/clean/v2", files=files)

    #     # Check the response
    #     if response.status_code == 200:
    #         data = response.json()
    #         print("Message:", data["message"])
    #         cleaned_df_req = data.get("cleaned_df") 
    #         if cleaned_df_req:
    #             cleaned_df = pd.read_json(cleaned_df_req)
    #             return cleaned_df
    #         else:
    #             print("No cleaned_df data found in the response.")
    #     else:
    #         print("Request to /clean/v2 failed with status code:", response.status_code)
    #         print("Response content:", response.content)
    # df=pd.read_csv(csv_filepath)
    df=clean_data(csv_filepath)
    return df



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
