from pathlib import Path

from loguru import logger
import pandas as pd

from .model import (
    SwiftModel,
    BankModel,
    add_finalreceiver_col,
    join_flags_to_swift_data,
)


def fit(swift_data_path: Path, bank_data_path: Path, model_dir: Path):
    swift_df = pd.read_csv(swift_data_path, index_col="MessageId")
    bank_df = pd.read_csv(bank_data_path, dtype=pd.StringDtype())

    logger.info("Preparing train data...")
    swift_df = add_finalreceiver_col(swift_df)
    swift_df = join_flags_to_swift_data(
        swift_df=swift_df,
        bank_df=bank_df,
    )

    # Train SWIFT model
    logger.info("Fitting SWIFT model...")
    swift_model = SwiftModel()
    swift_model.fit(X=swift_df[["InstructedCurrency"]], y=swift_df["Label"])

    # Train Bank model
    logger.info("Fitting Bank model...")
    bank_model = BankModel()
    bank_model.fit(X=swift_df[["BeneficiaryFlags"]], y=swift_df["Label"])

    logger.info("...done fitting")

    swift_model.save(model_dir / "swift_model.joblib")
    bank_model.save(model_dir / "bank_model.joblib")


def predict(
    swift_data_path: Path,
    bank_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
):
    swift_df = pd.read_csv(swift_data_path, index_col="MessageId")
    bank_df = pd.read_csv(bank_data_path, dtype=pd.StringDtype())

    logger.info("Preparing test data...")
    swift_df = add_finalreceiver_col(swift_df)
    swift_df = join_flags_to_swift_data(
        swift_df=swift_df,
        bank_df=bank_df,
    )

    logger.info("Loading models...")
    swift_model = SwiftModel.load(model_dir / "swift_model.joblib")
    bank_model = SwiftModel.load(model_dir / "bank_model.joblib")

    logger.info("Predicting on test data...")
    swift_preds = swift_model.predict(swift_df[["InstructedCurrency"]])
    bank_preds = bank_model.predict(swift_df[["BeneficiaryFlags"]])

    # Fill in missing bank predictions
    bank_preds = bank_preds.reindex(swift_preds.index, fill_value=1.0)

    # General final predictions
    final_preds = swift_preds * bank_preds

    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df["Score"] = preds_format_df.index.map(final_preds)

    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")
