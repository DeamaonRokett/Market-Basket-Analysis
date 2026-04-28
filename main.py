"""
main.py
~~~~~~~
End-to-end Market Basket Analysis pipeline.

Usage
-----
    python main.py                            # default settings
    python main.py --min-support 0.03 --min-lift 1.2
    python main.py --level product_name --min-support 0.01
    python main.py --basket-size 6 --skip-plots
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess import load_data, basic_eda, build_synthetic_baskets, build_basket_matrix, add_time_features
from association_rules import run_apriori, filter_rules, format_rules, save_rules
from visualise import (
    plot_category_distribution,
    plot_revenue_by_category,
    plot_support_distribution,
    plot_rules_scatter,
    plot_lift_heatmap,
    plot_network_graph,
    plot_reorder_analysis,
    plot_monthly_trend,
)
from recommender import BasketRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Market Basket Analysis Pipeline")
    p.add_argument("--data", default="data/generated_data.csv")
    p.add_argument("--level", choices=["category", "product_name"], default="category")
    p.add_argument("--basket-size", type=int, default=5,
                   help="Target items per synthetic basket (default: 5)")
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--min-lift", type=float, default=1.0)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--top-k-recs", type=int, default=5)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load & EDA ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Load & Explore Data")
    logger.info("=" * 60)
    df = load_data(args.data)
    summary = basic_eda(df)
    df = add_time_features(df)

    # ── 2. Build Basket Matrix ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Build Synthetic Baskets & Matrix")
    logger.info("=" * 60)
    df = build_synthetic_baskets(df, basket_size=args.basket_size)
    basket = build_basket_matrix(df, level=args.level)

    # ── 3. Apriori + Rules ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Mine Association Rules (Apriori)")
    logger.info("=" * 60)
    frequent_itemsets, rules = run_apriori(basket, min_support=args.min_support)

    if rules.empty:
        logger.warning("No rules generated. Try lowering --min-support or --min-lift.")
        top_rules = rules
        fmt_rules = format_rules(rules)
    else:
        top_rules = filter_rules(rules, min_confidence=args.min_confidence, min_lift=args.min_lift)
        fmt_rules = format_rules(top_rules)
        output_rules_path = Path("outputs") / "association_rules.csv"
        save_rules(top_rules, output_rules_path)
        logger.info("\n── Top 10 Rules by Lift ──")
        print(fmt_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string(index=False))

    # ── 4. Visualisations ─────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("=" * 60)
        logger.info("STEP 4 — Generate Visualisations")
        logger.info("=" * 60)
        plot_category_distribution(df)
        plot_revenue_by_category(df)
        plot_support_distribution(frequent_itemsets)
        if not top_rules.empty:
            plot_rules_scatter(top_rules)
            plot_lift_heatmap(top_rules)
            if len(top_rules) >= 5:
                plot_network_graph(top_rules)
        plot_reorder_analysis(df)
        plot_monthly_trend(df)
        logger.info("All charts saved to outputs/")

    # ── 5. Demo Recommendations ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Recommendation Engine Demo")
    logger.info("=" * 60)
    if fmt_rules.empty:
        logger.warning("No rules available for recommendations.")
    else:
        rec_engine = BasketRecommender(fmt_rules, top_k=args.top_k_recs)
        demo_baskets = {
            "user_A": ["Electronics", "Books"],
            "user_B": ["Toys", "Baby"],
            "user_C": ["Sports", "Outdoors"],
            "user_D": ["Beauty", "Health"],
        }
        for uid, items in demo_baskets.items():
            recs = rec_engine.recommend(items)
            if recs.empty:
                logger.info(f"{uid} basket={items} → no recommendations found")
            else:
                logger.info(f"{uid} basket={items}")
                print(recs[["recommended_item", "lift", "confidence"]].to_string(index=False))
                print()

    logger.info("Pipeline complete ✓")


if __name__ == "__main__":
    main()
