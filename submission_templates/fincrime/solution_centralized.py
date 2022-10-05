from pathlib import Path


def fit(
    swift_data_path: Path,
    bank_data_path: Path,
    model_dir: Path,
) -> None:
    """Function that fits your model on the provided training data and saves
    your model to disk in the provided directory.

    Args:
        swift_data_path (Path): Path to CSV data file for the SWIFT transaction
            dataset.
        bank_data_path (Path): Path to CSV data file for the bank account
            dataset.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None
    """
    ...


def predict(
    swift_data_path: Path,
    bank_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> None:
    """Function that loads your model from the provided directory and performs
    inference on the provided test data. Predictions should match the provided
    format and be written to the provided destination path.

    Args:
        swift_data_path (Path): Path to CSV data file for the SWIFT transaction
            dataset.
        bank_data_path (Path): Path to CSV data file for the bank account
            dataset.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None

    """
    ...
