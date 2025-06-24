import json
import yaml
import argparse
from typing import Dict, Any

from fleviden.core.aggregators.weighted_average import WeightedAverage
from fleviden.core.arch.cen.server import Server
from fleviden.core.bridge.http import HTTP
from fleviden.core.debug.logger import Logger
from fleviden.core.flow.collector import Collector
from fleviden.core.flow.ender import Ender
from fleviden.core.flow.juncture import Juncture
from fleviden.core.flow.starter import Starter
from fleviden.core.interfaces import Interfaces
from fleviden.core.loaders.csv import CSV
from fleviden.core.pod.pod import Pod
from fleviden.core.trainers.keras import Keras

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_server_pods(config: Dict[str, Any], model_path: str) -> None:
    """
    Setup and configure all pods based on the configuration
    """
    # Extract global configuration
    debug = config.get('DEBUG', False)
    verbose = config.get('VERBOSITY', 0)
    rounds = config.get('ROUNDS', 10)

    # Extract server specific configuration
    server_config = config['server']
    server_id = server_config['ID']
    clients = server_config.get('CLIENTS', [])
    min_clients = server_config['MIN_CLIENTS']
    data_path = server_config['DATA_PATH']
    features = server_config['FEATURES']
    targets = server_config['TARGETS']
    pd_args = server_config.get('PD_ARGS', {})

    # Create pods
    pod_server = Server(num_rounds=rounds, min_clients=min_clients)
    pod_collector = Collector(count=min_clients, input_entry="weights", output_entry="weights")
    pod_aggregator = WeightedAverage(input_key="weights", output_key="weights")
    pod_http = HTTP(server_id)
    pod_csv = CSV(filepath=data_path, features=features, targets=targets, pd_args=pd_args)
    pod_keras = Keras(
        load_model_filepath=model_path,
        initializer="random_normal",
        metrics=["accuracy"],
    )

    # Flow control pods
    pod_starter = Starter([Interfaces.LOAD], [{}])
    pod_juncture_evaluation = Juncture(count=2, keep=True, combine=True)
    pod_juncture_start = Juncture(count=2, keep=True, combine=True)
    pod_ender = Ender()
    pod_logger = Logger(level="debug" if debug else "info")

    # Initialization flow
    pod_starter.link(Interfaces.LOAD, pod_juncture_start, Interfaces.DOCK_0)
    pod_juncture_start.link(Interfaces.FIRE, pod_keras, Interfaces.LOAD_MODEL)
    pod_keras.link(Interfaces.INITIALIZED_MODEL, pod_server, Interfaces.BROADCAST)

    # Client subscriptions
    pod_http.wait("/rest/subscribe")
    pod_http.link("/rest/subscribe", pod_server, Interfaces.SUBSCRIBE)
    pod_server.link(Interfaces.START_TRAINING, pod_juncture_start, Interfaces.DOCK_1)

    # Federated round flow
    pod_aggregator.link(Interfaces.AGGREGATED, pod_server, Interfaces.BROADCAST)
    pod_server.link(Interfaces.BROADCASTED, pod_http, "/multicast")
    pod_http.bridge_multicast(
        "/multicast", "/multicasted", endpoint="/rest/update-from-server"
    )

    pod_http.wait(Interfaces.REST_UPDATE_FROM_CLIENT)
    pod_http.link(Interfaces.REST_UPDATE_FROM_CLIENT, pod_server, Interfaces.UPDATE)
    pod_server.link(Interfaces.UPDATED, pod_collector, Interfaces.INPUT)
    pod_collector.link(Interfaces.OUTPUT, pod_aggregator, Interfaces.AGGREGATE)

    # Evaluation flow
    pod_aggregator.link(Interfaces.AGGREGATED, pod_csv, Interfaces.LOAD)
    pod_aggregator.link(Interfaces.AGGREGATED, pod_juncture_evaluation, Interfaces.DOCK_0)
    pod_csv.link(Interfaces.LOADED, pod_juncture_evaluation, Interfaces.DOCK_1)
    pod_juncture_evaluation.link(Interfaces.FIRE, pod_keras, Interfaces.EVALUATE)
    pod_keras.link(Interfaces.EVALUATED, pod_logger, Interfaces.SEND_INFO)

    # Finish flow
    pod_server.link(Interfaces.STOP_TRAINING, pod_ender, Interfaces.FINISH)

    # Logging and Debugging
    Pod.link_all(Interfaces.DEBUG, pod_logger, Interfaces.SEND_DEBUG)
    Pod.link_all(Interfaces.ERROR, pod_logger, Interfaces.SEND_ERROR)
    Pod.link_all(Interfaces.WARNING, pod_logger, Interfaces.SEND_WARNING)
    Pod.link_all(Interfaces.INFO, pod_logger, Interfaces.SEND_INFO, False)
    Pod.debug_mode(debug, verbose=verbose)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Fleviden Server Configuration')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration YAML file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model file'
    )

     parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the model file'
    )

    return parser.parse_args()

def main():
    """
    Main function to run the server
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup pods with the model_path argument
    setup_server_pods(config, args.model_path, args.data_path)
    
    # Start fleviden
    Pod.start()

if __name__ == "__main__":
    main()