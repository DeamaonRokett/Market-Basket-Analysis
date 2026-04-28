"""
association_rules.py
~~~~~~~~~~~~~~~~~~~~~
Runs Apriori and generates association rules with lift / confidence / support.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_apriori(
    basket: pd.DataFrame,
    min_support: float = 0.05,
    min_threshold: float = 1.0,
    metric: str = "lift",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mine frequent itemsets and generate association rules.

    Parameters
    ----------
    basket        : binary order × item DataFrame
    min_support   : minimum support threshold (0–1)
    min_threshold : minimum metric value for rule filtering
    metric        : metric used for pruning rules ('lift', 'confidence', etc.)

    Returns
    -------
    (frequent_itemsets, rules) DataFrames
    """
    logger.info(f"Running Apriori  min_support={min_support}  metric={metric}≥{min_threshold}")
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    logger.info(f"Generated {len(rules)} rules")
    return frequent_itemsets, rules


def filter_rules(
    rules: pd.DataFrame,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_rules: int = 50,
) -> pd.DataFrame:
    """Apply secondary filters and return top rules."""
    filtered = rules[
        (rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)
    ].head(max_rules)
    logger.info(f"Filtered to {len(filtered)} rules (confidence≥{min_confidence}, lift≥{min_lift})")
    return filtered


def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert frozensets to readable strings and round metrics."""
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    out["consequents"] = out["consequents"].apply(lambda x: ", ".join(sorted(x)))
    for col in ["support", "confidence", "lift", "leverage", "conviction"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


def save_rules(rules: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = format_rules(rules)
    fmt.to_csv(output_path, index=False)
    logger.info(f"Rules saved → {output_path}")


if __name__ == "__main__":
    from preprocess import load_data, build_basket_matrix
    RAW = Path(__file__).parents[1] / "data" / "generated_data.csv"
    df = load_data(RAW)
    basket = build_basket_matrix(df, level="category")
    fi, rules = run_apriori(basket, min_support=0.05)
    top = filter_rules(rules)
    print(format_rules(top).head(10).to_string())
