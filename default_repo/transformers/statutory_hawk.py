from InteroperabilityEnabler.utils.annotation_dataset import add_quality_annotations_to_df

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, entity_type_annotation,
              assessed_attrs=None, type=None, context_value=None,
              *args, **kwargs):
    """
    Add quality annotations to a DataFrame for either
    instance-level or attribute-level annotations (but not both).

    Args:
        data (DataFrame): The flattened NGSI-LD data.
        entity_type (str): The NGSI-LD entity type for quality annotations.
        assessed_attrs (list of str, optional): To annotate with quality information (if None, annotate entire instance).
        type (str, optional): The default `type` for the DataFrame rows if not already exist.
        context_value (str or list, optional): The value to assign to the `@context` column if it does not exist.

    Returns:
        Pandas DataFrame with additional quality annotation columns.
    """
    annotated_df = add_quality_annotations_to_df(
        df,
        entity_type=entity_type_annotation,
        assessed_attrs=assessed_attrs,
        type=type,
        context_value=context_value
    )

    return annotated_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'