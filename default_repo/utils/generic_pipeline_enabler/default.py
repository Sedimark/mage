import io
import os
import re
import json
import time
import requests
import datetime
import pandas as pd
import numpy as np
from datetime import timezone
from InteroperabilityEnabler.utils.data_formatter import data_formatter
from InteroperabilityEnabler.utils.data_mapper import data_mapper
from mage_ai.orchestration.triggers.api import trigger_pipeline
from mage_ai.orchestration.db.models.schedules import PipelineRun

IGNORE_KWARGS = [
    "env", "execution_date", "interval_end_datetime", "interval_seconds", 
    "interval_start_datetime", "interval_start_datetime_previous", "event", 
    "logger", "configuration", "context", "pipeline_uuid", "block_uuid", "repo_path"
]
    

def load_temporal_data(ngsi_ld_host: str, ngsi_ld_link_context: str, entity_id: str, start_time: str | None, end_time: str | None, attrs: str | None, dataset_id: str | None) -> dict:
    """
    Load temporal data from the NGSI-LD API.
    Args:
        ngsi_ld_host (str): The base URL of the NGSI-LD API.
        ngsi_ld_link_context (str): The link context for NGSI-LD.
        entity_id (str): The ID of the entity to query.
        start_time (str | None): The start time for the query in ISO 8601 format (e.g., "2022-11-16T07:00:00Z").
        end_time (str | None): The end time for the query in ISO 8601 format (e.g., "2022-11-16T08:00:00Z").
        attrs (str | None): The attributes to retrieve from the entity.
    Returns:
        dict: The response from the NGSI-LD API containing the temporal data.
    Raises:
        ValueError: If the start_time or end_time is not in the correct format.
    """
    if ngsi_ld_host is None:
        raise ValueError("ngsi_ld_host must be provided")
    if ngsi_ld_link_context is None:
        raise ValueError("ngsi_ld_link_context must be provided")
    if entity_id is None:
        raise ValueError("entity_id must be provided")
    
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
                       
    url = f"{ngsi_ld_host}/ngsi-ld/v1/temporal/entities/{entity_id}"

    # <https://sedimark.github.io/broker/jsonld-contexts/sedimark-helsinki-compound.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"
    headers = {"Link": f'<{ngsi_ld_link_context}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'}

    if start_time is not None:
        try:
            datetime.datetime.strptime(start_time, datetime_format)
        except ValueError:
            raise ValueError(f"start_time must be in format {datetime_format} actual values is {start_time}")
    
    if end_time is not None:
        try:
            datetime.datetime.strptime(end_time, datetime_format)
        except ValueError:
            raise ValueError(f"end_time must be in format {datetime_format} actual values is {end_time} ")

    params = None
    timerel = None
    if start_time is not None:
        if end_time is None:
            end_time = datetime.datetime.now(timezone.utc).strftime(datetime_format)
        timerel = "between" 

    params = {"timerel": timerel, "timeAt": start_time, "endTimeAt": end_time, "attrs": attrs}

    if dataset_id is not None:
        params["datasetId"] = dataset_id

    response = requests.get(url, headers=headers, params=params)

    print(f"Request URL: {response.url}, Headers: {headers}, Params: {params}")
    response.raise_for_status()

    return response.json()


def export_temporal_data(temporal_data: dict, ngsi_ld_host: str, ngsi_ld_link_context: str) -> dict:
    """
    Export temporal data to the NGSI-LD API.
    Args:
        temporal_data (dict): The temporal data to export.
        ngsi_ld_host (str): The base URL of the NGSI-LD API.
        ngsi_ld_link_context (str): The link context for NGSI-LD.
    Returns:
        dict: The response from the NGSI-LD API after exporting the data.
    Raises:
        ValueError: If the temporal_data is not a dictionary.
    """
    if not isinstance(temporal_data, dict):
        raise ValueError("temporal_data must be a dictionary")
    
    url = f"{ngsi_ld_host}/ngsi-ld/v1/temporal/entities"
    headers = {
        "Content-Type": "application/json",
        "Link": f'<{ngsi_ld_link_context}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }
    
    # Hardcoded
    for entry in temporal_data['flow']:
        if "hasQuality" in entry:
            del entry['hasQuality']

    json_data = json.dumps(temporal_data, indent=2)
    print(json_data)

    response = requests.post(url, headers=headers, data=json_data)
    response.raise_for_status()

    return True


def interoperability_enabler_to_df(temporal_data: dict) -> pd.DataFrame:
    """
    Convert temporal data from NGSI-LD into a DataFrame using the InteroperabilityEnabler lib.
    Args:
        temporal_data (dict): The temporal data from the NGSI-LD API.
    Returns:
        pd.DataFrame: A DataFrame containing the converted temporal data.
    """
    if not isinstance(temporal_data, dict):
        raise ValueError("temporal_data must be a dictionary")
    
    # metadata_columns = ['id', 'type']
    # metadata_values = {f'entity_{k}': v for k, v in temporal_data.items() if k in metadata_columns}

    # def is_uri(s):
    #     return isinstance(s, str) and re.match(r'^https?://', s) is not None

    # value_columns = {k: v for k, v in temporal_data.items() if k not in metadata_columns and isinstance(v, list)}

    # result_df = pd.DataFrame()
    # for k, v in value_columns.items():
    #     temp_df = data_to_dataframe(v)
    #     if is_uri(k):
    #         temp_df['schema_uri'] = k
    #         temp_df['schema'] = k.split('/')[-1].split('#')[-1]
    #     else:
    #         temp_df['schema'] = k
    #     for prop, value in metadata_values.items():
    #         temp_df[prop] = value

    #     result_df = pd.concat([result_df, temp_df], ignore_index=True)

    context_df, temporal_df = data_formatter(temporal_data)

    return context_df, temporal_df


def interoperability_enabler_to_ngsild(context_df: pd.DataFrame, temporal_df: pd.DataFrame) -> dict:
    """
    Flatten dataframe based on schema_uri with column names in format: schema_uri[index].column_name
    
    Args:
        data: DataFrame with schema_uri column
        
    Returns:
        DataFrame: Flattened dataframe with new column naming convention
    """
    # entity_id = data['entity_id'][0]
    # entity_type = data['entity_type'][0]

    # schema_col = 'schema_uri' if 'schema_uri' in data.columns else 'schema'

    # if schema_col not in data.columns:
    #     raise ValueError(f"DataFrame must contain {schema_col} column")
    
    # schema = data[schema_col][0]
    # cols_to_remove = ['entity_id', 'entity_type']

    # if schema_col == 'schema_uri':
    #     cols_to_remove.append('schema')

    # data = data.drop(cols_to_remove, axis=1)
    # grouped = data.groupby(schema_col)
    
    # flattened_data = {}
    
    # for schema, group in grouped:
    #     # Reset index to get sequential indexing within each group
    #     group_reset = group.reset_index(drop=True)
        
    #     # For each row in the group, create columns with the new naming convention
    #     for idx, (_, row) in enumerate(group_reset.iterrows()):
    #         for col_name, value in row.items():
    #             if col_name == schema_col:
    #                 continue
    #             new_col_name = f"{schema}[{idx}].{col_name}"
    #             flattened_data[new_col_name] = value
    
    # flattened_df = pd.DataFrame([flattened_data])
    # flattened_df['id'] = entity_id
    # flattened_df['type'] = entity_type

    # # converted_data = data_conversion(flattened_df)
    # # restored_data = restore_ngsi_ld_structure(converted_data)
    restored_data = data_mapper(context_df, temporal_df)
    return restored_data


def json_to_dataframe_with_metadata(data_dict: list) -> pd.DataFrame:
    """
    Converts the standard list-of-dictionaries output from a Mage block into a pandas DataFrame.
    Args:
        data_dict (list): A list of dictionaries, where each dictionary contains 'sample_data' with 'columns' and 'rows'.
    Returns:
        pd.DataFrame: A DataFrame constructed from the provided data.
    Raises:
        Warning: If the input data_dict is empty or not a list, a warning is printed and an empty DataFrame is returned.
    """
    if not data_dict or not isinstance(data_dict, list):
        print("Warning: Received empty or invalid data from child pipeline. Returning an empty DataFrame.")
        return pd.DataFrame()
        
    all_data = []
    for item in data_dict:
        sample_data = item.get('sample_data', {})
        columns = sample_data.get('columns', [])
        rows = sample_data.get('rows', [])
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            all_data.append(row_dict)
    
    return pd.DataFrame(all_data)


def truncate_output(output, max_chars: int) -> str:
    """
    Truncates a string representation of an object for cleaner log output.
    Args:
        output: The output to be truncated, can be any object.
        max_chars (int): The maximum number of characters to include in the output string.
    Returns:
        str: A string representation of the output, truncated if necessary.
    """
    if output is None:
        return "None"
    output_str = str(output)
    if len(output_str) > max_chars:
        return output_str[:max_chars] + f"... [truncated, total length: {len(output_str)} chars]"
    return output_str


def execute_pipeline_and_get_final_result(
    pipeline_uuid: str,
    data: pd.DataFrame = None,
    kwargs: dict = None,
    max_output_chars: int = 200,
    poll_interval: int = 15
) -> list:
    """
    Triggers a pipeline and polls for its completion before returning the final block's result.
    Args:
        pipeline_uuid (str): The UUID of the pipeline to be triggered.
        data (pd.DataFrame, optional): Data to be passed to the pipeline. Defaults to None.
        kwargs (dict, optional): Additional keyword arguments for the pipeline run. Defaults to None.
        max_output_chars (int, optional): Maximum characters to include in the output string. Defaults to 200.
        poll_interval (int, optional): Time in seconds to wait between status checks. Defaults to 15.
    Returns:
        list: The output from the final block of the pipeline run.
    Raises:
        Exception: If the pipeline run fails or the final block does not complete successfully.
    """
    if kwargs is None:
        Exception("Kwargs must be provided for the child pipeline run.")
        
    variables = extract_kwargs(kwargs)

    if data is not None:
        data_flatten = data.to_dict(orient='records')
        variables['data_flatten'] = data_flatten
    else:
        variables['data_flatten'] = []
    
    pipeline_run = trigger_pipeline(
        pipeline_uuid=pipeline_uuid,
        variables=variables,
        check_status=False,  
        verbose=True
    )
    
    print(f"Triggering child pipeline '{pipeline_uuid}'")
    print(f"Pipeline run ID: {pipeline_run.id} | Initial status: {pipeline_run.status}")
    print(f"Now polling for completion every {poll_interval} seconds...")
    
    last_time_check = time.time()
    while pipeline_run.status in ['initial', 'running', 'queued']:
        current_time = time.time()

        if pipeline_run.status == 'completed':
            print(f"Pipeline run {pipeline_run.id} completed successfully.")
            break
        elif pipeline_run.status in ['failed', 'cancelled']:
            raise Exception(f"Child pipeline run {pipeline_run.id} failed with status: {pipeline_run.status}")

        if current_time - last_time_check >= 10:
            print(f"Status check for pipeline run {pipeline_run.id}: {pipeline_run.status}")
            last_time_check = current_time

        time.sleep(1)
        pipeline_run.refresh()
    
    print(f"Polling finished. Final status of run {pipeline_run.id}: {pipeline_run.status}") 
    print("Run completed successfully. Fetching results from the final block...")
    
    final_block_run = pipeline_run.block_runs[-1]
    result = final_block_run.get_outputs()
    df = json_to_dataframe_with_metadata(result)

    print(f"Output retrieved: {truncate_output(result, max_chars=max_output_chars)}")
    print(f"Successfully created final DataFrame with {len(df)} rows.")

    return df


def extract_kwargs(kwargs: dict) -> dict:
    """
    Extracts relevant keyword arguments for the pipeline run.
    Args:
        kwargs (dict): The keyword arguments to extract from.
    Returns:
        dict: A dictionary containing the extracted keyword arguments.
    """
    return {k: v for k, v in kwargs.items() if k not in IGNORE_KWARGS}


def load_parent_data(kwargs: dict) -> pd.DataFrame:
    """
    Loads DataFrame data from the parent pipeline's keyword arguments.
    Args:
        kwargs (dict): The keyword arguments from the parent pipeline.
    Returns:
        pd.DataFrame: DataFrame of the parent pipeline.
    """
    data = kwargs.get("data_flatten", None)
    df = pd.DataFrame(data)

    return df

def export_parent_data(data: pd.DataFrame) -> dict:
    """
    Exports DataFrame data to the parent pipeline.
    Args:
        data (pd.DataFrame): The DataFrame to be exported.
    Returns:
        dict: The return data for the parent pipeline.
    """
    return data.to_dict(orient='records')