import pika
import json
from typing import Any
from mage_ai.data_preparation.decorators import data_loader
from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_file(*args, **kwargs):
    """
    Template for loading data from filesystem.
    Load data from 1 file or multiple file directories.

    For multiple directories, use the following:
        FileIO().load(file_directories=['dir_1', 'dir_2'])

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    print("Making the connection!")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="62.72.21.79", port=5673)
    )

    queue_name = 'hello'
    channel.queue_declare(queue='hello')

    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
    
    if not method_frame:
        return {}

    return json.loads(body)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
