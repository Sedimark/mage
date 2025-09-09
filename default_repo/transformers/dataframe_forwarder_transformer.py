import json
from sklearn.preprocessing import StandardScaler

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    scaler = StandardScaler()

    cols = [col for col in data.columns if col.endswith("__value")]
    data[cols] = scaler.fit_transform(data[cols])
    
    scaler_config = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "n_features_in_": scaler.n_features_in_
    }

    with open("default_repo/scaler_config.json", "w") as f:
        json.dump(scaler_config, f)

    return data.to_dict(orient='records')


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'