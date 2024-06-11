import pika
import json
import time
import asyncio
import websockets
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

    async def send_data(self, data):
        uri = "ws://62.72.21.79:30719/mage/ws"
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(data))

            time.sleep(2)

    async def consume_queue(self, queue_name: str, handler: Callable):
        """
        Asynchronous method to consume messages from the queue and send them via WebSocket.
        """
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=False)
            if method_frame:
                self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                try:
                    data = json.loads(body)
                    await self.send_data(data)
                    handler([data])
                except json.JSONDecodeError:
                    self.channel.close()
                    self.connection.close()
                    break

    def batch_read(self, handler: Callable):
        """
        Batch read the messages from the source and use handler to process the messages.
        """
        queue_name = 'hello'
        self.channel.queue_declare(queue='hello')

        asyncio.run(self.consume_queue(queue_name, handler))
