from InteroperabilityEnabler.utils.data_mapper import data_conversion, restore_ngsi_ld_structure

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def data_mapper(annotated_df, *args, **kwargs):
    """
        Convert a DataFrame into NGSI-LD json format.

        Args:
            df (DataFrame): The input DataFrame (from CSV, XLS/XLSX or flattened NGSI-LD JSON).

        Returns:
            A NGSI-LD data.
    """
    data = data_conversion(annotated_df)
    data_restored = restore_ngsi_ld_structure(data)

    return data_restored


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
