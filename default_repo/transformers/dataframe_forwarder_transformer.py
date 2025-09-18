import pickle
from crossformer.utils.tools import Preprocessor

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    cols = [col for col in data.columns if col.endswith("__value")]
    preprocessor = Preprocessor(method="zscore",per_feature=True)
    preprocessor.fit(data[cols].values)
    data[cols] = preprocessor.transform(data[cols].values)
    stats = preprocessor.export()

    with open("default_repo/scaler_config.pkl", "wb") as f:
        pickle.dump(stats, f)

    return data.to_dict(orient='records')


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'