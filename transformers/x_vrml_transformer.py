import xml.etree.ElementTree as ET

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
    tree = ET.ElementTree(ET.fromstring(data))
    root = tree.getroot()

    translation_vector = [2, 3, 1]

    for coordinate in root.findall(".//Coordinate"):
        points = coordinate.get('point').split(',')
        new_points = []
        for point in points:
            x, y, z = map(float, point.split())
            x += translation_vector[0]
            y += translation_vector[1]
            z += translation_vector[2]
            new_points.append(f"{x} {y} {z}")
        coordinate.set('point', ', '.join(new_points))

    data = ET.tostring(root, encoding='unicode')

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'