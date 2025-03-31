from default_repo.utils.inria.annotation_dataset import add_quality_annotations_to_df

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Transformer block for data annotation component

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    type = kwargs.get("type")
    context = kwargs.get("context") if kwargs.get("context") else ""
    entity_type = kwargs.get("entity_type")

    if not type or not entity_type:
        raise Exception("Needed information to run the block is not provided!")

    data = add_quality_annotations_to_df(data, entity_type=entity_type, assessed_attrs=None, type=type, context_value=context)

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
