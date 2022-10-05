from pathlib import Path
from typing import Tuple, Union

import flwr as fl


class TrainClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for training."""


def train_client_factory(
    cid: str,
    data_path: Path,
    client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        data_path (Path): Path to CSV data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    ...
    return TrainClient(...)


class TrainStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for training."""


def train_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    ...
    return TrainStrategy(...)


class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""


def test_client_factory(
    cid: str,
    data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages. The
            SWIFT node will always be named 'swift'.
        data_path (Path): Path to CSV test data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Optional[Path]): Path to CSV file matching the format
            you must write your predictions with, filled with dummy values. This
            will only be non-None for the 'swift' client—bank clients should not
            write any predictions and receive None for this argument.
        preds_dest_path (Optional[Path]): Destination path that you must write
            your test predictions to as a CSV file. This will only be non-None
            for the 'swift' client—bank clients should not write any predictions
            and will receive None for this argument.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    ...
    return TestClient(...)


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""


def test_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    ...
    return TestStrategy(...), 1
