from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd

from .model import (
    SwiftModel,
    BankModel,
    add_finalreceiver_col,
    join_flags_to_swift_data,
)


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return fl.common.ndarrays_to_parameters([])


# TRAIN PROCEDURE:
# round 1:
#   - SWIFT client fits on SWIFT data; SWIFT client sends labels for banks to Strategy
#   - Bank clients tell Strategy which banks are present in each partition
# round 2:
#   - Strategy sends labels to banks; banks join flag data and fit


def swift_df_to_ndarrays(
    swift_df: pd.DataFrame, labels: bool = True
) -> List[np.ndarray]:
    """Utility function that converts a pandas DataFrame of SWIFT data to a list of
    numpy arrays, which the expected format for communication between a Flower
    NumPyClient and a Flower Strategy."""
    cols = [
        "FinalReceiver",
        "BeneficiaryAccount",
    ]
    if labels:
        cols.append("Label")
    # Need .astype("U") because pandas uses 'object' for string columns by default
    # 'object' arrays are not serializable without pickle.
    return [
        # Index
        swift_df.index.values.astype("U"),
        # Transactions: Join keys and label
        swift_df[cols].values.astype("U"),
    ]


def ndarrays_to_swift_df(
    swift_index: np.ndarray, swift_transactions: np.ndarray, labels: bool = True
) -> pd.DataFrame:
    """Utility function that converts a list of numpy arrays, which the expected format
    for communication between a Flower NumPyClient and a Flower Strategy, back to a
    pandas DataFrame"""
    cols = [
        "FinalReceiver",
        "BeneficiaryAccount",
    ]
    if labels:
        cols.append("Label")
    return pd.DataFrame(
        data=swift_transactions,
        index=pd.Series(swift_index, name="MessageId"),
        columns=cols,
    )


class TrainingSwiftClient(fl.client.NumPyClient):
    def __init__(
        self, cid: str, swift_df: pd.DataFrame, model: SwiftModel, client_dir: Path
    ):
        super().__init__()
        self.cid = cid
        self.swift_df = swift_df
        self.model = model
        self.client_dir = client_dir

    def fit(self, parameters: List[np.ndarray], config: dict):
        # Fit model
        logger.info(f"{self.cid} : Fitting model...")
        X = self.swift_df[["InstructedCurrency"]]
        y = self.swift_df["Label"]
        self.model.fit(X, y)
        # Save model checkpoint
        logger.info(f"{self.cid} : Saving model to disk...")
        self.model.save(self.client_dir / "model.joblib")

        # Send labels through server to bank client(s)
        swift_df = add_finalreceiver_col(self.swift_df)
        labels_for_banks: List[np.ndarray] = swift_df_to_ndarrays(swift_df)
        return labels_for_banks, X.shape[0], {}


class TrainingBankClient(fl.client.NumPyClient):
    def __init__(
        self, cid: str, bank_df: pd.DataFrame, model: BankModel, client_dir: Path
    ):
        super().__init__()
        self.cid = cid
        self.bank_df = bank_df
        self.model = model
        self.client_dir = client_dir

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        # Round 1: Send banks in this partition to strategy
        if config["round"] == 1:
            return [self.bank_df["Bank"].unique().astype("U")], 0, {}
        # Round 2: Fit NB model on data
        elif config["round"] == 2:
            logger.info(f"{self.cid} : Received SWIFT labels...")
            # parameters contains label data from SWIFT, reform into dataframe
            swift_index, swift_transactions = parameters
            swift_df = ndarrays_to_swift_df(swift_index, swift_transactions)
            # Join ordering account flags
            logger.info(f"{self.cid} : Joining bank flags to SWIFT labels...")
            swift_df = join_flags_to_swift_data(swift_df, self.bank_df)
            # Fit model
            logger.info(f"{self.cid} : Fitting model...")
            X = swift_df[["BeneficiaryFlags"]]
            y = swift_df["Label"]
            self.model.fit(X, y)
            # Save model checkpoint
            logger.info(f"{self.cid} : Saving model to disk...")
            self.model.save(self.client_dir / "model.joblib")
            return [], X.shape[0], {}
        else:
            raise Exception(f"Unexpected round {config['round']}")


def train_client_factory(cid, data_path: Path, client_dir: Path):
    if cid == "swift":
        logger.info("Initializing SWIFT client for {}", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        model = SwiftModel()
        return TrainingSwiftClient(
            cid, swift_df=swift_df, model=model, client_dir=client_dir
        )
    else:
        logger.info("Initializing bank client for {}", cid)
        bank_df = pd.read_csv(data_path, dtype=pd.StringDtype())
        model = BankModel()
        return TrainingBankClient(
            cid, bank_df=bank_df, model=model, client_dir=client_dir
        )


class TrainStrategy(fl.server.strategy.Strategy):
    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        self.swift_labels_for_banks = None
        self.banks_dict = {}
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        if server_round == 1:
            fit_config: List[Tuple[ClientProxy, FitIns]] = []
            # SWIFT fits on SWIFT data
            swift_fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            fit_config += [(client_dict["swift"], swift_fit_ins)]
            # Banks report which banks are present
            bank_fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            fit_config += [
                (v, bank_fit_ins) for k, v in client_dict.items() if k != "swift"
            ]
            return fit_config
        elif server_round == 2:
            # Configure bank clients to fit on labels sent from SWIFT
            # Turn stashed swift labels into dataframe
            swift_index, swift_transactions = self.swift_labels_for_banks
            swift_df = ndarrays_to_swift_df(
                swift_index=swift_index, swift_transactions=swift_transactions
            )
            # Banks get sent labels to fit models
            bank_cids = [cid for cid in client_dict.keys() if cid != "swift"]
            fit_config = []
            for cid in bank_cids:
                # Subset swift labels to banks present for this client
                this_bank_swift_df = swift_df[
                    swift_df["FinalReceiver"].isin(self.banks_dict[cid])
                ]
                # Convert dataframe to parameters
                this_bank_fit_ins = FitIns(
                    parameters=fl.common.ndarrays_to_parameters(
                        swift_df_to_ndarrays(this_bank_swift_df)
                    ),
                    config=config_dict,
                )
                fit_config.append((client_dict[cid], this_bank_fit_ins))
            return fit_config

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures
    ) -> Tuple[Optional[Parameters], dict]:
        if (n_failures := len(failures)) > 0:
            raise Exception(f"Had {n_failures} failures in round {server_round}")
        if server_round == 1:
            for client, result in results:
                result_ndarrays = fl.common.parameters_to_ndarrays(result.parameters)
                if client.cid == "swift":
                    # This is SWIFT client's results. Stash for later
                    self.swift_labels_for_banks = result_ndarrays
                else:
                    # This is a bank client. Stash which banks are present
                    self.banks_dict[client.cid] = result_ndarrays[0]
        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Not running any federated evaluation."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Not aggregating any evaluation."""
        return None

    def evaluate(self, server_round, parameters):
        """Not running any centralized evaluation."""
        return None


def train_strategy_factory(server_dir: Path):
    training_strategy = TrainStrategy(server_dir=server_dir)
    num_rounds = 2
    return training_strategy, num_rounds


# TEST PROCEDURE:
# round 1:
#   - SWIFT client sends accounts for banks to Strategy
#   - Bank clients tell Strategy which banks are present in each partition
# round 2:
#   - Strategy sends accounts to Banks; Banks predict and send predictions back
# round 3:
#   - Strategy sends Bank predictions to SWIFT. SWIFT predicts and combines.


def test_client_factory(
    cid: str,
    data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
):
    if cid == "swift":
        logger.info("Initializing SWIFT client for {}", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        return TestSwiftClient(
            cid,
            swift_df=swift_df,
            client_dir=client_dir,
            preds_format_path=preds_format_path,
            preds_dest_path=preds_dest_path,
        )
    else:
        logger.info("Initializing bank client for {}", cid)
        bank_df = pd.read_csv(data_path, dtype=pd.StringDtype())
        return TestBankClient(cid, bank_df=bank_df, client_dir=client_dir)


class TestSwiftClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        swift_df: pd.DataFrame,
        client_dir: Path,
        preds_format_path: Path,
        preds_dest_path: Path,
    ):
        super().__init__()
        self.cid = cid
        self.swift_df = swift_df
        self.client_dir = client_dir
        self.preds_format_path = preds_format_path
        self.preds_dest_path = preds_dest_path

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        round = config["round"]
        if round == 1:
            # Send account columns through server to bank clients
            swift_df = add_finalreceiver_col(self.swift_df)
            return swift_df_to_ndarrays(swift_df, labels=False), 0, {}
        elif round == 3:
            # Predict with SWIFT model, join with bank predictions
            # then write to disk
            model = SwiftModel.load(self.client_dir / "model.joblib")
            X = self.swift_df[["InstructedCurrency"]]
            swift_preds = model.predict(X)
            # Join bank predictions
            bank_indices, bank_preds = parameters
            bank_preds = pd.Series(data=bank_preds, index=bank_indices, name="Score")
            # Fill in missing bank predictions
            bank_preds = bank_preds.reindex(swift_preds.index, fill_value=1.0)
            # Calculate final predictions
            final_preds = swift_preds * bank_preds
            # Read format, write predictions to destination
            preds_format_df = pd.read_csv(self.preds_format_path, index_col="MessageId")
            preds_format_df["Score"] = preds_format_df.index.map(final_preds)
            preds_format_df.to_csv(self.preds_dest_path)
            return [], 0, {}
        else:
            raise Exception(f"Unexpected round {round} for SWIFT client.")


class TestBankClient(fl.client.NumPyClient):
    def __init__(self, cid, bank_df, client_dir):
        super().__init__()
        self.cid = cid
        self.bank_df = bank_df
        self.client_dir = client_dir

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        ## Round 1: Send banks in this partition to strategy
        if config["round"] == 1:
            return [self.bank_df["Bank"].unique().astype("U")], 0, {}
        ## Round 2: Load NB model, predict on data
        elif config["round"] == 2:
            logger.info(f"{self.cid} : Loading model...")
            model = BankModel.load(self.client_dir / "model.joblib")
            logger.info(f"{self.cid} : Received SWIFT transactions...")
            # parameters contains account data from SWIFT, reform into dataframe
            swift_index, swift_transactions = parameters
            swift_df = ndarrays_to_swift_df(
                swift_index, swift_transactions, labels=False
            )
            # Join ordering account flags
            logger.info(f"{self.cid} : Joining bank flags to SWIFT transactions...")
            swift_df = join_flags_to_swift_data(swift_df, self.bank_df)
            # Run predict
            logger.info(f"{self.cid} : Predicting...")
            X = swift_df[["BeneficiaryFlags"]]
            preds = model.predict(X)
            return (
                [preds.index.values.astype("U"), preds.values],
                X.shape[0],
                {},
            )
        else:
            raise Exception(f"Unexpected round {config['round']}")


def test_strategy_factory(server_dir: Path):
    test_strategy = TestStrategy(server_dir=server_dir)
    num_rounds = 3
    return test_strategy, num_rounds


class TestStrategy(fl.server.strategy.Strategy):
    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        self.swift_transactions_for_banks = None
        self.banks_dict = {}
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        if server_round == 1:
            ## All clients have something to do, don't need any data sent
            fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            return [(client, fit_ins) for client in client_dict.values()]
        elif server_round == 2:
            ## Send swift transactions with account info to bank clients
            # Turn stashed swift labels into dataframe
            swift_index, swift_transactions = self.swift_transactions_for_banks
            swift_df = ndarrays_to_swift_df(
                swift_index=swift_index,
                swift_transactions=swift_transactions,
                labels=False,
            )
            # Banks get sent transaction account info to predict on
            bank_cids = [cid for cid in client_dict.keys() if cid != "swift"]
            fit_config = []
            for cid in bank_cids:
                # Subset swift labels to banks present for this client
                this_bank_swift_df = swift_df[
                    swift_df["FinalReceiver"].isin(self.banks_dict[cid])
                ]
                # Convert dataframe to parameters
                this_bank_fit_ins = FitIns(
                    parameters=fl.common.ndarrays_to_parameters(
                        swift_df_to_ndarrays(this_bank_swift_df, labels=False)
                    ),
                    config=config_dict,
                )
                fit_config.append((client_dict[cid], this_bank_fit_ins))
            return fit_config
        elif server_round == 3:
            ## Send bank predictions back to SWIFT
            fit_ins = FitIns(
                parameters=fl.common.ndarrays_to_parameters(
                    [self.bank_indices, self.bank_preds]
                ),
                config=config_dict,
            )
            return [(client_dict["swift"], fit_ins)]

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures
    ) -> Tuple[Optional[Parameters], dict]:
        """Do not fit during test."""
        if (n_failures := len(failures)) > 0:
            raise Exception(f"Had {n_failures} failures in round {server_round}")
        if server_round == 1:
            for client, result in results:
                result_ndarrays = fl.common.parameters_to_ndarrays(result.parameters)
                if client.cid == "swift":
                    # This is SWIFT client's results. Stash for later
                    self.swift_transactions_for_banks = result_ndarrays
                else:
                    # This is a bank client. Stash which banks are present
                    self.banks_dict[client.cid] = result_ndarrays[0]
        elif server_round == 2:
            # Banks sent back predictions
            # result.parameters is (index, preds)
            client_bank_indices, client_bank_preds = zip(
                *(
                    fl.common.parameters_to_ndarrays(result.parameters)
                    for _, result in results
                )
            )
            # Stash to send back to SWIFT client
            self.bank_indices = np.concatenate(client_bank_indices)
            self.bank_preds = np.concatenate(client_bank_preds)
        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Not running any federated evaluation."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Not aggregating any evaluation."""
        return None

    def evaluate(self, server_round: int, parameters: fl.common.typing.Parameters):
        """Not running any centralized evaluation."""
        return None
