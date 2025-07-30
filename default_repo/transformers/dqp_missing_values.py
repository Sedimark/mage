from dqp import DataSource, MissingImputationModule
import logging

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
    
    data_source = DataSource(data.copy())

    mehod = kwargs.get("method", "KNNImputer")

    config = {
        "imputation_method": method
    }

    logger.info("Running configuration is %s.", config)

    module = MissingImputationModule(**config)

    result = module(data_source)._df

    return result


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'