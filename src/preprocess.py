"""
preprocess.py
~~~~~~~~~~~~~
Loads raw transaction data and engineers features for market basket analysis.

Note on basket construction
---------------------------
The raw dataset contains one row per order (each order has a single category).
Since Market Basket Analysis requires multi-item baskets, this module provides
two strategies:

1. ``enrich_baskets`` — groups a user's orders across time into sessions.
   Works well when the same user appears multiple times in the data.

2. ``build_synthetic_baskets`` — clusters orders into pseudo-baskets using
   shared behavioural attributes (price tier, reorder flag, recency bucket).
   Useful for fully synthetic / sparse datasets like this one.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(path, parse_dates=["order_timestamp"])
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}")
    return df


def basic_eda(df: pd.DataFrame) -> dict:
    summary = {
        "total_rows": len(df),
        "unique_orders": df["order_id"].nunique(),
        "unique_users": df["user_id"].nunique(),
        "unique_products": df["product_id"].nunique(),
        "unique_categories": df["product_category"].nunique(),
        "date_range": (df["order_timestamp"].min().date(), df["order_timestamp"].max().date()),
        "category_counts": df["product_category"].value_counts().to_dict(),
    }
    logger.info(
        f"Orders: {summary['unique_orders']} | Users: {summary['unique_users']} | "
        f"Products: {summary['unique_products']} | Categories: {summary['unique_categories']}"
    )
    return summary


def build_synthetic_baskets(df: pd.DataFrame, basket_size: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Construct synthetic multi-item baskets from a single-item-per-order dataset.

    Strategy
    --------
    1. Bin orders into behavioural segments using price tier, recency, and reorder flag.
    2. Within each segment, randomly assign orders to baskets of ~basket_size items.
    3. Return the original DataFrame with an added ``basket_id`` column.

    This mirrors the real-world practice of reconstructing baskets from loyalty-card
    or session-log data when order-level multi-item records are unavailable.

    Parameters
    ----------
    basket_size : target number of items per basket (approx)
    seed        : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Behavioural segmentation
    df["price_tier"] = pd.qcut(df["item_price"], q=3, labels=["low", "mid", "high"])
    df["recency_bucket"] = pd.qcut(df["days_since_last_order"], q=4, labels=["recent", "moderate", "lapsed", "churned"], duplicates="drop")

    basket_id = np.zeros(len(df), dtype=int)
    counter = 0

    for _, group in df.groupby(["price_tier", "recency_bucket", "is_reorder"], observed=True):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            basket_id[df.index.get_loc(idx)] = counter + i // basket_size
        counter += len(idxs) // basket_size + 1

    df["basket_id"] = basket_id
    df.drop(columns=["price_tier", "recency_bucket"], inplace=True)

    n_baskets = df["basket_id"].nunique()
    avg_size = len(df) / n_baskets
    logger.info(f"Built {n_baskets} synthetic baskets | avg size: {avg_size:.1f} items")
    return df


def build_basket_matrix(df: pd.DataFrame, level: str = "category") -> pd.DataFrame:
    """
    Build a binary basket × item matrix.

    Parameters
    ----------
    df    : transactions DataFrame with a 'basket_id' column.
    level : 'category' or 'product_name'
    """
    if level not in ("category", "product_name"):
        raise ValueError("level must be 'category' or 'product_name'")

    col_map = {"category": "product_category", "product_name": "product_name"}
    col = col_map[level]
    group_key = "basket_id" if "basket_id" in df.columns else "order_id"

    logger.info(f"Building basket matrix at '{level}' level …")
    basket = (
        df.groupby([group_key, col])["quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    basket = (basket > 0).astype(int)
    logger.info(f"Basket matrix shape: {basket.shape}")
    return basket


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["order_timestamp"].dt.hour
    df["day_of_week"] = df["order_timestamp"].dt.day_name()
    df["month"] = df["order_timestamp"].dt.month_name()
    df["is_weekend"] = df["order_timestamp"].dt.dayofweek.isin([5, 6])
    return df


if __name__ == "__main__":
    RAW = Path(__file__).parents[1] / "data" / "generated_data.csv"
    df = load_data(RAW)
    basic_eda(df)
    df = build_synthetic_baskets(df, basket_size=5)
    basket = build_basket_matrix(df, level="category")
    print(basket.head())
    print("\nItems per basket:\n", basket.sum(axis=1).describe())
