import pickle
import pandas as pd
from crossformer.utils.tools import Postprocessor

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


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
    value_cols = [col for col in data.columns if col.endswith("__value")]
    
    with open("default_repo/scaler_config.pkl", "rb") as fp:
        stats = pickle.load(fp)

    postprocessor = Postprocessor(stats=stats)

    data[value_cols] = pd.DataFrame(postprocessor.inverse_transform(data[value_cols].values))

    return data.to_dict(orient='records')