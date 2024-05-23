import pika
import json
from mage_ai.streaming.sources.base_python import BasePythonSource
from typing import Callable

if 'streaming_source' not in globals():
    from mage_ai.data_preparation.decorators import streaming_source


@streaming_source
class CustomSource(BasePythonSource):
    def init_client(self):
        """
        Implement the logic of initializing the client.
        """
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="62.72.21.79", port=5673)
        )
        self.channel = self.connection.channel()

    def batch_read(self, handler: Callable):
        """
        Batch read the messages from the source and use handler to process the messages.
        """
        while True:
            records = []
            queue_name = 'hello'
            self.channel.queue_declare(queue='hello')

            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=False)

            if method_frame:
                self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                try:
                    records.append(json.loads(body))
                except json.decoder.JSONDecodeError:
                    self.channel.close() 
                    self.connection.close()
                    break
                    
            if len(records) > 0:
                handler(records)
