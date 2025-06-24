import yaml
import argparse
from typing import Dict, Any

from fleviden.core.arch.cen.client import Client
from fleviden.core.bridge.http import HTTP
from fleviden.core.debug.logger import Logger
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

def setup_client_pods(config: Dict[str, Any], model_path: str, model_path: str) -> None:
    """
    Setup and configure all pods based on the configuration
    """
    # Extract global configuration
    debug = config.get('DEBUG', False)
    verbose = config.get('VERBOSITY', 0)
    rounds = config.get('ROUNDS', 10)

    # Extract client-specific configuration
    client_config = config['client']
    client_id = client_config['ID']
    server = client_config['SERVER']
    epochs = client_config['EPOCHS']
    batch_size = client_config['BATCH_SIZE']
    data_path = client_config['DATA_PATH']
    features = client_config['FEATURES']
    targets = client_config['TARGETS']
    pd_args = client_config.get('PD_ARGS', {})

    server_host = f"http://{server}"

    # Create pods
    pod_client = Client(client_id, server, num_rounds=rounds)
    pod_keras = Keras(
        load_model_filepath=model_path,
        epochs=epochs,
        batch_size=batch_size,
        send_gradients=False
    )
    pod_csv = CSV(
        filepath=data_path,
        features=features,
        targets=targets,
        pd_args=pd_args
    )
    pod_http = HTTP(client_id)

    # Flow control pods
    pod_starter = Starter(["/send-subscription-request"], [{}])
    pod_juncture = Juncture(count=2, keep=True, combine=True)
    pod_ender = Ender()
    pod_logger = Logger(level="debug" if debug else "info")

    # Client subscription
    pod_starter.link("/send-subscription-request", pod_client, Interfaces.SUBSCRIBE)
    pod_client.link(Interfaces.SUBSCRIBED, pod_http, "/send-subscription-request")
    pod_http.bridge(
        "/send-subscription-request",
        host=server_host,
        endpoint="/rest/subscribe",
    )

    # Federated round flow
    pod_http.wait("/rest/update-from-server")
    pod_http.link("/rest/update-from-server", pod_client, Interfaces.UPDATE)
    pod_client.link(Interfaces.FORWARDED, pod_http, Interfaces.UPDATE)
    pod_http.bridge(
        Interfaces.UPDATE,
        host=server_host,
        endpoint=Interfaces.REST_UPDATE_FROM_CLIENT
    )

    # Train flow
    pod_client.link(Interfaces.UPDATED, pod_juncture, Interfaces.DOCK_0)
    pod_csv.link(Interfaces.LOADED, pod_juncture, Interfaces.DOCK_1)
    pod_client.link(Interfaces.UPDATED, pod_csv, Interfaces.LOAD)
    pod_juncture.link(Interfaces.FIRE, pod_keras, Interfaces.TRAIN)
    pod_keras.link(Interfaces.TRAINED, pod_client, Interfaces.FORWARD)

    # Finish flow
    pod_client.link(Interfaces.COMPLETED, pod_ender, Interfaces.FINISH)

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
    parser = argparse.ArgumentParser(description='Fleviden Client Configuration')
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
    Main function to run the client
    """
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Setup pods with the model_path argument
    setup_client_pods(config, args.model_path, args.data_path)

    # Start fleviden
    Pod.start()

if __name__ == "__main__":
    main()