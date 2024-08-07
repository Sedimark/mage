import trimesh
import numpy as np
from io import StringIO

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
    mesh_io = StringIO(data)

    mesh = trimesh.load(mesh_io, file_type='obj')

    mirror_matrix = np.array([
        [-1,  0,  0, 0],
        [ 0,  1,  0, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1]
    ])

    mesh.apply_transform(mirror_matrix)

    mesh_string = mesh.export(file_type='obj')

    return mesh_string


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'