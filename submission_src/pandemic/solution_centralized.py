from pathlib import Path


def fit(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
) -> None:
    """Function that fits your model on the provided training data and saves
    your model to disk in the provided directory.

    Args:
        person_data_path (Path): Path to CSV data file for the Person table.
        household_data_path (Path): Path to CSV data file for the House table.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table.
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
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> None:
    """Function that loads your model from the provided directory and performs
    inference on the provided test data. Predictions should match the provided
    format and be written to the provided destination path.

    Args:
        person_data_path (Path): Path to CSV data file for the Person table.
        household_data_path (Path): Path to CSV data file for the House table.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table.
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
