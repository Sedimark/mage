from mage_ai.orchestration.triggers.api import trigger_pipeline
import time
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def json_to_dataframe_with_metadata(data_dict):
    """
    Convert with full metadata preservation
    """
    all_data = []
    
    for item in data_dict:
        sample_data = item.get('sample_data', {})
        columns = sample_data.get('columns', [])
        rows = sample_data.get('rows', [])
        
        # Convert each row to a dictionary
        for row in rows:
            row_dict = dict(zip(columns, row))
            all_data.append(row_dict)
    
    return pd.DataFrame(all_data)


def truncate_output(output, max_chars):
       """Truncate output to specified character limit"""
       if output is None:
           return None
       
       output_str = str(output)
       if len(output_str) <= max_chars:
           return output_str
       else:
           return output_str[:max_chars] + f"... [truncated, total length: {len(output_str)} chars]"


def execute_pipeline_and_get_final_result(pipeline_uuid, variables=None, max_output_chars=10):
    """
    Execute a pipeline and return the final block's result
    """
    
    if variables is None:
        variables = {}
    
    # Trigger the pipeline
    pipeline_run = trigger_pipeline(
        pipeline_uuid=pipeline_uuid,
        variables=variables,
        check_status=True,
        verbose=True
    )
    
    print(f"Pipeline run ID: {pipeline_run.id}")
    print(f"Initial status: {pipeline_run.status}")
    
    # Monitor pipeline execution
    while pipeline_run.status in ['running', 'queued']:
        time.sleep(5)
        pipeline_run.refresh()
        print(f"Status: {pipeline_run.status}")
    
    print(f"Final status: {pipeline_run.status}")
    
    result = None
    for block_run in pipeline_run.block_runs:
        block_uuid = block_run.block_uuid
        block_info = {
            'status': block_run.status,
            'logs': None,
            'output': None
        }
        
        try:
            if hasattr(block_run, 'get_logs'):
                block_info['logs'] = block_run.get_logs()
            elif hasattr(block_run, 'logs'):
                block_info['logs'] = block_run.logs
        except Exception as e:
            block_info['logs'] = f"Error retrieving logs: {e}"
        
        # Get output
        try:
            if hasattr(block_run, 'get_outputs'):
                block_info['output'] = block_run.get_outputs()
            elif hasattr(block_run, 'output'):
                block_info['output'] = block_run.output
        except Exception as e:
            block_info['output'] = f"Error retrieving output: {e}"
        
        print(f"\n--- Block: {block_uuid} ---")
        print(f"Status: {block_info['status']}")
        print(f"Logs: {truncate_output(block_info['logs'], max_chars=max_output_chars)}")
        print(f"Output: {truncate_output(block_info['output'], max_chars=max_output_chars)}")

        if pipeline_run.block_runs[-1] == block_run and block_info['status'] == 'completed':
            result = block_info['output']

    return result


@transformer
def transform(data, *args, **kwargs):
    ignore_kwargs = ["env", "execution_date", "interval_end_datetime", "interval_seconds", "interval_start_datetime", "interval_start_datetime_previous", "event", "logger", "configuration", "context", "pipeline_uuid", "block_uuid", "repo_path"]
    # pipeline_uuid = 'data_preprocessing_test'
    # pipeline_uuid = kwargs.get("data_preprocessing_pipeline_trigger", None)
    pipeline_uuid = kwargs.get("data_preprocessing_pipeline_trigger", "data_preprocessing_test")

    data_flatten = data.to_dict(orient='records')
    variables = {'data_flatten': data_flatten}

    for entry in kwargs:
        if entry in ignore_kwargs:
            continue
        variables[entry] = kwargs.get(entry, None)

    result = execute_pipeline_and_get_final_result(pipeline_uuid, variables, max_output_chars=200)
    df = json_to_dataframe_with_metadata(result)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'