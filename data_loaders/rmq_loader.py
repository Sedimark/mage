import pika
import json
import threading

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
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="62.72.21.79", port=5673)
    )

    channel = connection.channel()
    queue_name = 'hello'
    channel.queue_declare(queue='hello')

    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=False)

    if method_frame:
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
        try:
            return_body = json.loads(body)
            return return_body
        except json.decoder.JSONDecodeError: 
            return {}
    
    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'