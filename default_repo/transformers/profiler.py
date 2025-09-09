import sys
import logging
import tempfile
import subprocess
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

logger = logging.getLogger(__name__)


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        method (str): Can be one of: 'SimpleImputer', 'KNNImputer', 'LogisticRegression' or 'Interpolation'. Default is KNNImputer.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    logger.info("Running configuration is %s.", {})

    data = pd.read_csv("/home/src/default_repo/broker_values.csv")
    value_columns = [column for column in data.columns if column.endswith("__value")]

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmp:
        data.to_csv(tmp.name, index=False)

        result = subprocess.run(
            ["bash", "/home/src/default_repo/utils/dqp_scripts/dqp_profiling.sh", tmp.name],
            shell=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        print("Return code:", result.returncode)
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        
        if result.returncode != 0:
            print(f"Script failed with code {result.returncode}")
            return data.to_csv("record")

        result = pd.read_csv(tmp.name)

    data.loc[:, data.columns] = result.values

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'