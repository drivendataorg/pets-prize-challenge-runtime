from pathlib import Path
from typing import Tuple, Union
import utils
import warnings

import flwr as fl
from fedProx import FedProx


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.model_selction import train_test_split


class TrainClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for training."""
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = LogisticRegression(
                penalty="l2",
                max_iter=1,  # local epoch
                warm_start=True,  # prevent refreshing weights when fitting
                )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    
    def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        self.model = utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        self.model = utils.set_model_params(self.model, parameters)
        
        y_score = self.model.predict_proba(self.X_train)[:, 1]
        # Average precision score
        average_precision = average_precision_score(self.y_train, y_score)
        print(average_precision)
        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_train, self.y_score)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        print("Model Training Accuracy Metric", auc_precision_recall)
      
        y_score = self.model.predict_proba(self.X_test)[:, 1]
        # Average precision score
        average_precision = average_precision_score(self.y_test, y_score)
        print(average_precision)
        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_score)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        print("Model Testing Accuracy Metric",auc_precision_recall)      
        return auc_precision_recall, len(self.X_test), {"accuracy": average_precision}


def train_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    ...
    data = utils.get_model_data('path')
    X = data.drop(['pid', 'covid'],axis = 1).values
    y = data['covid']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.1,
                                                        random_state=0,
                                                        stratify=y)
    
    return TrainClient(x_train, y_train, x_test, y_test)


class TrainStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for training."""
    def __init__(self):
        self.strategy = FedProx()
        
    def initialize_parameters(self, client_manager):
        return self.strategy.initialize_parameters(client_manager)
    def configure_fit(self, server_round, parameters, client_manager):
        return self.strategy.configure_fit(server_round, parameters, client_manager)
    def aggregate_fit(self,server_round,results,failures):
        return self.strategy.aggregate_fit(server_round, results, failures)
    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)
    def aggregate_evaluate(self,server_round,results,failures):
        return self.strategy.aggregate_evaluate(server_round, results, failures)
    def evaluate(self, server_round, parameters):
        return self.strategy.evaluate(server_round, parameters)
    

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

    return TrainStrategy()


class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""
    def __init__(self, X_train, y_train):
        self.model = LogisticRegression(
                penalty="l2",
                max_iter=1,  # local epoch
                warm_start=True,  # prevent refreshing weights when fitting
                )
        self.X_train = X_train
        self.y_train = y_train
        
    
    def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
    
        return None, len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        self.model = utils.set_model_params(self.model, parameters)
        
        y_score = self.model.predict_proba(self.X_train)[:, 1]
        # Average precision score
        average_precision = average_precision_score(self.y_train, y_score)
        print(average_precision)
        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_train, self.y_score)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        print("Model Training Accuracy Metric", auc_precision_recall)
      
        return auc_precision_recall, len(self.X_train), {"accuracy": precision}


def test_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
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
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    data = utils.get_model_data('path')
    X = data.drop(['pid', 'covid'],axis = 1).values
    y = data['covid']
    return TestClient(X,y)


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""
    def __init__(self):
        self.strategy = FedProx()
        
    def initialize_parameters(self, client_manager):
        return self.strategy.initialize_parameters(client_manager)
    def configure_fit(self, server_round, parameters, client_manager):
        return self.strategy.configure_fit(server_round, parameters, client_manager)
    def aggregate_fit(self,server_round,results,failures):
        return self.strategy.aggregate_fit(server_round, results, failures)
    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)
    def aggregate_evaluate(self,server_round,results,failures):
        return self.strategy.aggregate_evaluate(server_round, results, failures)
    def evaluate(self, server_round, parameters):
        return self.strategy.evaluate(server_round, parameters)


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
    return TestStrategy(), 1
