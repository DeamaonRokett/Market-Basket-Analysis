"""
test_pipeline.py
~~~~~~~~~~~~~~~~
Unit tests for the Market Basket Analysis pipeline.

Run: pytest tests/test_pipeline.py -v
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from preprocess import load_data, basic_eda, build_synthetic_baskets, build_basket_matrix, add_time_features
from association_rules import run_apriori, filter_rules, format_rules
from recommender import BasketRecommender


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_data(Path(__file__).parents[1] / "data" / "generated_data.csv")


@pytest.fixture(scope="module")
def enriched_df(raw_df):
    return build_synthetic_baskets(raw_df, basket_size=5)


@pytest.fixture(scope="module")
def basket(enriched_df):
    return build_basket_matrix(enriched_df, level="category")


@pytest.fixture(scope="module")
def mined(basket):
    return run_apriori(basket, min_support=0.05)


@pytest.fixture(scope="module")
def top_rules(mined):
    _, rules = mined
    return filter_rules(rules, min_confidence=0.0, min_lift=1.0)


@pytest.fixture(scope="module")
def fmt_rules(top_rules):
    return format_rules(top_rules)


# ── Data Loading Tests ─────────────────────────────────────────────────────────

class TestDataLoading:
    def test_loads_correct_shape(self, raw_df):
        assert raw_df.shape[0] == 1000
        assert raw_df.shape[1] == 10

    def test_required_columns_present(self, raw_df):
        required = {"order_id", "user_id", "product_category", "quantity", "item_price"}
        assert required.issubset(raw_df.columns)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")

    def test_eda_summary_keys(self, raw_df):
        summary = basic_eda(raw_df)
        assert "unique_orders" in summary
        assert "unique_categories" in summary
        assert summary["unique_orders"] == raw_df["order_id"].nunique()


# ── Preprocessing Tests ────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_synthetic_baskets_created(self, enriched_df):
        assert "basket_id" in enriched_df.columns
        assert enriched_df["basket_id"].nunique() > 1

    def test_avg_basket_size_reasonable(self, enriched_df):
        avg = enriched_df.groupby("basket_id").size().mean()
        assert 2 <= avg <= 10

    def test_basket_matrix_binary(self, basket):
        assert basket.isin([0, 1]).all().all()

    def test_basket_level_category(self, enriched_df):
        b = build_basket_matrix(enriched_df, level="category")
        assert set(b.columns).issubset(set(enriched_df["product_category"].unique()))

    def test_basket_invalid_level(self, raw_df):
        with pytest.raises(ValueError):
            build_basket_matrix(raw_df, level="invalid")

    def test_time_features_added(self, raw_df):
        enriched = add_time_features(raw_df)
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in enriched.columns

    def test_hour_range(self, raw_df):
        enriched = add_time_features(raw_df)
        assert enriched["hour"].between(0, 23).all()


# ── Association Rules Tests ────────────────────────────────────────────────────

class TestAssociationRules:
    def test_frequent_itemsets_not_empty(self, mined):
        fi, _ = mined
        assert len(fi) > 0

    def test_support_within_bounds(self, mined):
        fi, _ = mined
        assert fi["support"].between(0, 1).all()

    def test_rules_columns_present(self, mined):
        _, rules = mined
        for col in ["support", "confidence", "lift", "antecedents", "consequents"]:
            assert col in rules.columns

    def test_filter_reduces_rules(self, mined):
        _, rules = mined
        filtered = filter_rules(rules, min_confidence=0.5, min_lift=1.5)
        assert len(filtered) <= len(rules)

    def test_format_rules_produces_strings(self, fmt_rules):
        if not fmt_rules.empty:
            assert pd.api.types.is_string_dtype(fmt_rules["antecedents"])
            assert pd.api.types.is_string_dtype(fmt_rules["consequents"])


# ── Recommender Tests ──────────────────────────────────────────────────────────

class TestRecommender:
    def test_returns_dataframe(self, fmt_rules):
        rec = BasketRecommender(fmt_rules, top_k=5)
        result = rec.recommend(["Electronics"])
        assert isinstance(result, pd.DataFrame)

    def test_top_k_limit(self, fmt_rules):
        rec = BasketRecommender(fmt_rules, top_k=3)
        result = rec.recommend(["Toys"])
        assert len(result) <= 3

    def test_no_duplicates_in_recommendations(self, fmt_rules):
        rec = BasketRecommender(fmt_rules, top_k=10)
        result = rec.recommend(["Baby"])
        assert result["recommended_item"].is_unique

    def test_empty_basket_returns_empty(self, fmt_rules):
        rec = BasketRecommender(fmt_rules)
        result = rec.recommend([])
        assert result.empty

    def test_unknown_item_returns_empty(self, fmt_rules):
        rec = BasketRecommender(fmt_rules)
        result = rec.recommend(["NonexistentXYZ_999"])
        assert result.empty

    def test_batch_recommend_keys(self, fmt_rules):
        rec = BasketRecommender(fmt_rules)
        baskets = {"u1": ["Electronics"], "u2": ["Sports"]}
        out = rec.batch_recommend(baskets)
        assert set(out.keys()) == {"u1", "u2"}
        for df in out.values():
            assert isinstance(df, pd.DataFrame)

    def test_empty_rules_graceful(self):
        empty_rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
        rec = BasketRecommender(empty_rules)
        result = rec.recommend(["Electronics"])
        assert result.empty
