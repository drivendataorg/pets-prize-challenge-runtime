import json
from pathlib import Path

import numpy as np
import pandas as pd


class SirModel:
    """A simplistic risk model based on the discrete-time version of the SIR disease
    model. This model estimates the parameter beta from the SIR equations, which is the
    contact rate of infected individuals of a population.

    Given a disease outcome time series for the population, the fit method will estimate
    beta from the data. The model stores the running sums in the numerator and
    denominator to allow combining with estimates from other datasets.

    Attributes:
        lookahead (int): Lookahead window in days. Conceptually, the timestep of the
            SIR difference equation. Defaults to 7.
        numerator (float): Numerator of beta estimate.
        denominator (float): Denominator of average beta estimate.
    """

    def __init__(
        self,
        lookahead: int = 7,
        numerator: float = None,
        denominator: float = None,
    ):
        self.lookahead: int = lookahead
        if (numerator is None or denominator is None) and not (
            numerator is None and denominator is None
        ):
            raise Exception(
                "Cannot instantiate model with only one of numerator or denominator "
                "fitted."
            )
        self.numerator: float = numerator
        self.denominator: float = denominator

    @property
    def beta(self) -> float:
        """Estimated beta value from data."""
        if self.numerator is None:
            return None
        return self.numerator / self.denominator

    def fit(self, disease_outcome_df: pd.DataFrame):
        """Estimates the beta parameter of the SIR Model.

        Args:
            disease_outcome_df (pd.DataFrame): Disease outcome time series dataframe
            lookahead (int): Number of days in the future to predict an infection event
        """
        from loguru import logger

        # Get "infection events" for when individuals transition from S to I
        # This should be the first I state in time, generally
        # Case where individual is I on first day doesn't matter, because it won't get
        # counted in any lookahead windows later
        logger.info("Identifying infection events...")
        infection_events_df = (
            disease_outcome_df[disease_outcome_df["state"] == "I"]
            .sort_values(["day", "pid"])
            .drop_duplicates("pid", keep="first")
        )

        # Get S, I, R counts per day
        logger.info("Counting disease states per day...")
        day_df = disease_outcome_df.groupby("day").agg(
            S=("state", lambda ser: (ser == "S").sum()),
            I=("state", lambda ser: (ser == "I").sum()),
            N=("pid", "size"),
        )

        def count_infection_events(day: int):
            window_events_df = infection_events_df[
                (day < infection_events_df["day"])
                & (infection_events_df["day"] <= day + self.lookahead)
            ]
            return window_events_df.shape[0]

        # Get count of new infections over the lookahead period
        logger.info("Counting infection events in lookahead window...")
        day_df["next_infections"] = day_df.index.map(count_infection_events)

        day_df = day_df.iloc[: -self.lookahead]

        y = day_df["next_infections"]
        x = day_df["I"] * day_df["S"] / day_df["N"]

        # Calculate numerator and denominator for beta estimator
        self.numerator = np.dot(x.values, y.values)
        self.denominator = np.dot(x.values, x.values)

    def predict(self, disease_outcome_df: pd.DataFrame) -> pd.Series:
        if self.beta is None:
            raise Exception("Can't run predict with unfitted model.")

        # population size
        n = disease_outcome_df["pid"].nunique()

        # get state counts on last day of disease outcome data
        last_day = disease_outcome_df.loc[
            disease_outcome_df.day == disease_outcome_df.day.max()
        ]
        last_day_i = (last_day["state"] == "I").sum()  # num infected

        # new_infections = beta * i * s / n
        # probability = new_infections / s
        probability_of_infection = self.beta * last_day_i / n

        return pd.Series(
            data=[probability_of_infection] * n,
            name="score",
            index=pd.Index(data=last_day["pid"].values, name="pid"),
        )

    def set_params(self, numerator: float, denominator: float):
        self.numerator = numerator
        self.denominator = denominator

    def save(self, model_path: Path):
        """Save model to disk."""
        with model_path.open("w") as fp:
            json.dump(
                {
                    "lookahead": self.lookahead,
                    "numerator": self.numerator,
                    "denominator": self.denominator,
                },
                fp,
            )

    @classmethod
    def load(cls, model_path: Path) -> "SirModel":
        """Load model from disk."""
        with model_path.open("r") as fp:
            params = json.load(fp)
        model = cls(**params)
        return model
