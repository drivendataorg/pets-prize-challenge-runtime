import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB


class SwiftModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("encoder", OrdinalEncoder()),
                ("model", CategoricalNB()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst


def add_finalreceiver_col(swift_data: pd.DataFrame):
    """Adds column identifying FinalReciver to SWIFT dataset inplace. Required for
    joining to the bank data.

    See https://www.drivendata.org/competitions/105/nist-federated-learning-2-financial-crime-federated/page/589/#end-to-end-transactions
    """
    uetr_groups_train = swift_data.sort_values("Timestamp").groupby("UETR")
    swift_data["FinalReceiver"] = swift_data["UETR"].map(
        uetr_groups_train.Receiver.last().to_dict()
    )
    return swift_data


def join_flags_to_swift_data(swift_df: pd.DataFrame, bank_df: pd.DataFrame):
    """Join BeneficiaryFlags columns onto SWIFT dataset."""
    # Join beneficiary account flags
    swift_df = (
        swift_df.reset_index()
        .merge(
            right=bank_df[["Bank", "Account", "Flags"]].rename(
                columns={"Flags": "BeneficiaryFlags"}
            ),
            how="left",
            left_on=["FinalReceiver", "BeneficiaryAccount"],
            right_on=["Bank", "Account"],
        )
        .set_index("MessageId")
    )
    return swift_df


class BankModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=pd.NA, strategy="constant", fill_value="-1"
                    ),
                ),
                ("encoder", OrdinalEncoder()),
                ("model", CategoricalNB()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if len(self.pipeline.named_steps["model"].classes_) == 1:
            # Training data only had class 0
            return pd.Series([0.0] * X.shape[0], index=X.index)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst
