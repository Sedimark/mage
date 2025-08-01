from default_repo.utils.generic_pipeline_enabler.default import export_parent_data
from dqp import DataSource, DeduplicationModule
import pandas as pd
import logging

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def convert_data_type_to_record_linkage(data: pd.DataFrame, column: str) -> str:
    dtype = data[column].dtype
    
    if pd.api.types.is_numeric_dtype(dtype):
        return "number"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    else:
        return "string"


logger = logging.getLogger(__name__)


@transformer
def transform(data, *args, **kwargs):
    """
    Handles deduplication using the DQP module.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        method (str): RecordLinkage or ActiveDedupe. Default ActiveDedupe
        columns (list, None): List of columns that will be used for RecordLinkage. If empty all of them will be used. ActiveDedupa does the inference automatically.
        indexing_method (str): Full, Block or Neighbourhood. Needed for RecordLinkage. Default Full
        index_column (str): The column that will be used as index when doing the linkage rules.
        match_threshold (str): How many matched columns are required to determine if the rows are matches.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    data_source = DataSource(data.copy())

    columns = kwargs.get("columns")
    method = kwargs.get("method")
    indexing_method = kwargs.get("indexing_method", "Full")
    index_column = kwargs.get("index_column")
    match_threshold = kwargs.get("match_threshold", 2)

    config = {
        "method": method,
        "processing_options": "remove"
    }

    if method == "RecordLinkage":
        parameters["model_config"]: {
            "linkage_rules": [],
            "match_threshold": match_threshold,
            "indexing_method": indexing_method,
            "index_column": index_column,
        }

        if isinstance(columns, list):
            if index_column:
                config["model_config"]["linkage_rules"] = [
                    {
                        "field_1": column,
                        "field_2": column,
                        "base_method": convert_data_type_to_record_linkage(data, column) 
                        "parameters": {},
                    }
                    for column in columns if column != index_column and column in data.columns
                ]
            else:
                config["model_config"]["linkage_rules"] = [
                    {
                        "field_1": column,
                        "field_2": column,
                        "base_method": convert_data_type_to_record_linkage(data, column) 
                        "parameters": {},
                    }
                    for column in columns if column in data.columns
                ]
        else:
            if index_column:
                config["model_config"]["linkage_rules"] = [
                    {
                        "field_1": column,
                        "field_2": column,
                        "base_method": convert_data_type_to_record_linkage(data, column) 
                        "parameters": {},
                    }
                    for column in data.columns if column != index_column
                ]
            else:
                config["model_config"]["linkage_rules"] = [
                    {
                        "field_1": column,
                        "field_2": column,
                        "base_method": convert_data_type_to_record_linkage(data, column) 
                        "parameters": {},
                    }
                    for column in data.columns
                ]

    logger.info("Running configuration is %s.", config)

    module = DeduplicationModule(**config)

    result = module.process(data_source)._df

    return export_parent_data(result)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'