from datetime import datetime
from typing import Tuple

import pandas as pd

def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    targets_columns_names: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    """
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(targets_columns_names, axis=1)
    y_train = train_data[targets_columns_names]
    X_test = test_data.drop(targets_columns_names, axis=1)
    y_test = test_data[targets_columns_names]

    return X_train, y_train, X_test, y_test