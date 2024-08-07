import os
import base64
import trimesh

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    
    file_data = kwargs.get("file")

    if file_data is None:
        raise ValueError("file_data kwargs is missing!")

    binary_data = base64.b64decode(file_data["value"])

    file_path = os.path.join("objects", file_data["name"])

    with open(file_path, 'wb') as file:
        file.write(binary_data)

    mesh = trimesh.load(f'/home/src/default_repo/objects/{file_data["name"]}')

    mesh_string = mesh.export(file_type='obj')

    return mesh_string


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'