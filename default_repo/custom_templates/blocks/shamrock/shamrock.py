from mage_ai.streaming.sources.base_python import BasePythonSource
from mage_ai.settings.repo import get_repo_path
from websockets.sync.client import connect
from typing import Callable, Any
from threading import Event
from queue import Queue
import requests
import asyncio
import yaml
import json

if 'streaming_source' not in globals():
    from mage_ai.data_preparation.decorators import streaming_source


@streaming_source
class CustomSource(BasePythonSource):
    def init_client(self):
        """
        Implement the logic of initializing the client.
        """
        self.stop_event = Event()
        self.update_queue = Queue()
        config_file = get_repo_path() + "/configs/<pipeline_name>/config.yaml"
        self.accuracies = []
        self.active_peers = []
        self.done = False
        self.past_accuracies = self.accuracies
        with open(config_file, 'r') as c:
            self.conf = yaml.safe_load(c)
        
        if self.conf["dataset"].get("entity_id"):
            self.load_ngsi_ld(self.conf["dataset"].get("entity_id"))
        
        if self.conf["model"].get("model_uri"):
            self.save_model_locally(self.conf["model"].get("model_uri"))

        self.get_node()

        self.generator = self.topology.fit_generator(self.node)
        self.epoch = 0
        self.metric_name = self.conf["model"]["metrics"][0]
        self.websocket_url = "ws://mageapi:8000/mage/ws"
    
    def save_model_locally(self, model_uri: str):
        response = requests.request("GET", model_uri)

        file_name = response.headers.get("content-disposition").split("=")[-1]
        model_path = get_repo_path() + "/configs/<pipeline_name>/model_files/" + file_name
        with open(model_path, 'wb') as fp:
            for chunk in response.iter_content(chunk_size=8192):
                fp.write(chunk)
        
        self.conf["model"]["model_path"] = model_path


    def load_ngsi_ld(self, entity_id: str):
        host = os.getenv("NGSI_LD_HOST")
        link_context = os.getenv("NGSI_LD_LINK_CONTEXT")
        tenant = os.getenv("NGSI_LD_TENANT")
        if not host or not link_context or not entity_id:
            raise Exception("Needed information to run the block is not provided!")

        bucket = {
            'host': host,
            'entity_id': entity_id,
            'link_context': link_context,
            'time_query': f'timerel=after&timeAt=2022-01-01T00:00:00Z',
        }

        stellio_broker = connector.DataStore_NGSILD(bucket['host'])

        load_data = connector.LoadData_NGSILD(
            data_store=stellio_broker, 
            entity_id=bucket['entity_id'],
            context=bucket['link_context'],
            tenant=tenant,
        )

        load_data.run(bucket)

        data = bucket['temporal_data']
        data.reset_index(inplace=True)

        train = data.sample(frac=0.7, random_state=42)

        test = data.drop(train.index)

        train_path = get_repo_path() + "/configs/<pipeline_name>/train.csv"
        test_path = get_repo_path() + "/configs/<pipeline_name>/test.csv"
        train.to_csv(get_repo_path() + "/configs/<pipeline_name>/train.csv", index=False)
        test.to_csv(get_repo_path() + "/configs/<pipeline_name>/test.csv", index=False)

        self.conf["dataset"]["train"] = train_path
        self.conf["dataset"]["test"] = test_path

    def send_data(self, data: Any):
        """
        Sends data through the WebSocket connection.
        """
        try:
            with connect(self.websocket_url) as websocket:
                print(f"Seding data to websocket: {data}")
                websocket.send(json.dumps(data), text=True)
        except Exception as e:
            print(f"Error sending data: {e}")

    def get_node(self):
        from shamrock import (
            ConfigLoader,
            ModelLoader,
            TopologyLoader,
            ShamrockDataset,
            ShamrockNode,
        )
        from shamrock.config import Config
        from shamrock.model.builtin import builtin_model

        if self.conf["model"].get("model"):
            model_class = builtin_model[self.conf["framework"]][self.conf["model"]["model"]]
            model = model_class()
            self.conf["model"]["model"] = model

        from shamrock.utils.condition import (
            BasicStopCondition,
            FederatedServerStopCondition,
        )

        if self.conf["topology"]["topology_name"] != "FederatedServer":

            condition = BasicStopCondition(stop_event=self.stop_event)
        else:
            condition = FederatedServerStopCondition(stop_event=self.stop_event)
        config_object = Config(**self.conf)
        data = ShamrockDataset(**config_object.dataset)
        model = ModelLoader(**config_object.model)
        topology = TopologyLoader(stop_condition=condition, **config_object.topology)
        node = ShamrockNode(dataset=data, model=model, **config_object.node)
        self.node = node
        self.topology = topology

    def batch_read(self, handler: Callable):
        """
        Batch read the messages from the source and use handler to process the messages.
        """
        while True:
            try:
                for success, metric in self.generator:

                    if success and metric[self.metric_name] is not None:
                        self.active_peers = [peer._address for peer in self.node.active_peers()]

                        self.update_queue.put(
                            {
                                "type": "loss",
                                "data": {
                                    "loss": metric[self.metric_name],
                                    "epoch": self.epoch,
                                },
                            }
                        )

                        while not self.update_queue.empty():
                            update = self.update_queue.get()
                            if update["type"] == "loss":
                                self.accuracies.append(update["data"]["loss"])
                            elif update["type"] == "error":
                                print(f"Error: {update['data']}", "Training error occurred")
                        
                        handler(self.update_queue)

                        send_to_ws = {
                            "pipeline": "<pipeline_name>",
                            "type": "json",
                            "data": self.accuracies,
                            "done": self.node._status,
                            "peers": self.active_peers
                        }

                        self.send_data(send_to_ws)

                if self.node._status != "active" and not self.done:
                    self.done = True
                    send_to_ws = {
                        "pipeline": "<pipeline_name>",
                        "type": "json",
                        "data": self.accuracies,
                        "done": self.node._status,
                        "peers": self.active_peers
                    }

                    self.send_data(send_to_ws)
                    
            except Exception as e:
                import traceback

                print(traceback.format_exc())
                self.update_queue.put({"type": "error", "data": str(e)})
