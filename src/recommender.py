"""
recommender.py
~~~~~~~~~~~~~~
Rule-based product recommendation engine powered by association rules.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BasketRecommender:
    """
    Recommend items based on what is already in a user's basket.

    Parameters
    ----------
    rules : formatted rules DataFrame (antecedents and consequents as strings).
            Pass the output of ``format_rules()`` directly.
    top_k : maximum number of recommendations to return per basket.
    """

    def __init__(self, rules: pd.DataFrame, top_k: int = 5):
        self.top_k = top_k
        self.rules = rules.copy()
        if self.rules.empty:
            return

        # Normalise: convert frozensets → comma-separated strings if needed
        first_ant = self.rules["antecedents"].iloc[0]
        if not isinstance(first_ant, str):
            self.rules["antecedents"] = self.rules["antecedents"].apply(
                lambda x: ", ".join(sorted(x))
            )
            self.rules["consequents"] = self.rules["consequents"].apply(
                lambda x: ", ".join(sorted(x))
            )

    def recommend(self, basket_items: list) -> pd.DataFrame:
        """
        Given a list of items in the current basket, return ranked recommendations.

        Parameters
        ----------
        basket_items : list of category / product names already in basket.

        Returns
        -------
        DataFrame with columns [recommended_item, lift, confidence, support, rule].
        """
        if self.rules.empty or not basket_items:
            return pd.DataFrame(
                columns=["recommended_item", "lift", "confidence", "support", "rule"]
            )

        basket_set = set(basket_items)
        matches = []

        for _, row in self.rules.iterrows():
            antecedent_items = {a.strip() for a in row["antecedents"].split(",")}
            if antecedent_items.issubset(basket_set):
                consequent_items = {c.strip() for c in row["consequents"].split(",")}
                new_items = consequent_items - basket_set
                for item in new_items:
                    matches.append(
                        {
                            "recommended_item": item,
                            "lift": row["lift"],
                            "confidence": row["confidence"],
                            "support": row["support"],
                            "rule": f"{row['antecedents']} → {row['consequents']}",
                        }
                    )

        if not matches:
            logger.info(f"No rules matched basket: {basket_items}")
            return pd.DataFrame(
                columns=["recommended_item", "lift", "confidence", "support", "rule"]
            )

        recs = (
            pd.DataFrame(matches)
            .sort_values("lift", ascending=False)
            .drop_duplicates("recommended_item")
            .head(self.top_k)
            .reset_index(drop=True)
        )
        return recs

    def batch_recommend(self, baskets: dict) -> dict:
        """
        Recommend for multiple users at once.

        Parameters
        ----------
        baskets : {user_id: [item1, item2, …]}
        Returns a dict of {user_id: recommendations_df}.
        """
        return {uid: self.recommend(items) for uid, items in baskets.items()}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess import load_data, build_synthetic_baskets, build_basket_matrix
    from association_rules import run_apriori, filter_rules, format_rules

    RAW = Path(__file__).parents[1] / "data" / "generated_data.csv"
    df = load_data(RAW)
    df = build_synthetic_baskets(df, basket_size=5)
    basket = build_basket_matrix(df, level="category")
    fi, rules = run_apriori(basket, min_support=0.05)
    top_rules = filter_rules(rules)
    fmt_rules = format_rules(top_rules)

    rec = BasketRecommender(fmt_rules, top_k=5)
    print("\n--- Demo ---")
    sample = ["Electronics", "Books"]
    print(f"Basket: {sample}")
    print(rec.recommend(sample).to_string(index=False))
