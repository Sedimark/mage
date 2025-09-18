from default_repo.utils.crossformer_wrap.manipulation import initialize_manipulation, MageCrossFormer
from crossformer.utils.tools import Postprocessor
import pandas as pd
import pickle

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
    cfg_base = {
    # "data_dim": 8,
    "in_len": 96,
    "out_len": 24,
    "seg_len": 2,
    "window_size": 4,
    "factor": 10,
    "model_dim": 256,
    "feedforward_dim": 512,
    "head_num": 4,
    "layer_num": 6,
    "dropout": 0.2,
    "baseline": False,
    "learning_rate": 0.1,
    "batch_size": 8,
    "split": [0.7, 0.2, 0.1],
    "seed": 2024,
    "accelerator": "auto",
    "min_epochs": 1,
    "max_epochs": 2,
    "precision": 32,
    "patience": 5,
    "num_workers": 31,
    "method": "zscore",
    }
    # cfg_base = {
    #     # "data_dim": 8,
    #     "in_len": 12,
    #     "out_len": 6,
    #     "seg_len": 2,
    #     "window_size": 4,
    #     "factor": 10,
    #     "model_dim": 256,
    #     "feedforward_dim": 512,
    #     "head_num": 4,
    #     "layer_num": 6,
    #     "dropout": 0.2,
    #     "baseline": False,
    #     "learning_rate": 0.1,
    #     "batch_size": 2,
    #     "split": [0.6, 0.3, 0.1],
    #     "seed": 2024,
    #     "accelerator": "auto",
    #     "min_epochs": 1,
    #     "max_epochs": 2,
    #     "precision": 32,
    #     "patience": 5,
    #     "num_workers": 31,
    # }

    value_cols = [col for col in data.columns if col.endswith("__value")]
    print(f'Value columns: {value_cols}')
    print(data[value_cols].shape)

    with open("default_repo/scaler_config.pkl", "rb") as fp:
        stats = pickle.load(fp)

    postprocessor = Postprocessor(stats=stats)

    data[value_cols] = pd.DataFrame(postprocessor.inverse_transform(data[value_cols].values))
    
    manipulation = initialize_manipulation("mage_crossformer", cfg=cfg_base)
    model_name = manipulation.train(data[value_cols])
    
    return model_name, value_cols, cfg_base['in_len'], data