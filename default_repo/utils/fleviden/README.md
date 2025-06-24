# Fleviden example

## Description

This is a basic example to run a federated learning scenario with Fleviden. The scenario comprises:

- Two clients
- One server

The script that implements a client is implemented at `scripts/client.py`, while the server is implemented at `scripts/server.py`. The scripts simply import Fleviden **pods** and interconnect them to achieve the desired functionality. Fleviden scripts then create and run on an async loop that is triggered with the `Pod.start()` method.

The example uses:
- **Keras** as training framework.
- **HTTP** as communication protocol between client and server.
- **CSV** as dataloader for the clients' data.
- **FedAvg** as the aggregation technique.


## How to use

1. Setup the scenario by modifying the configuration file `config.yaml`. Here you can indicate the number of rounds, local training epochs, and other values of interest.

2. Open a terminal
3. Execute:

```
make run
```
or the equivalent

```
docker compose up --build
```
