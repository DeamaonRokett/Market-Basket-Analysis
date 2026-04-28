"""
visualise.py
~~~~~~~~~~~~
All plotting functions for the Market Basket Analysis project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

PALETTE = "viridis"
FIG_DPI = 150
OUT_DIR = Path(__file__).parents[1] / "outputs"


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / name
    fig.savefig(fp, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {fp}")
    return fp


# ─── 1. Category distribution ────────────────────────────────────────────────

def plot_category_distribution(df: pd.DataFrame) -> Path:
    counts = df["product_category"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette(PALETTE, len(counts)))
    ax.set_title("Transaction Count per Product Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Transactions")
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _save(fig, "01_category_distribution.png")


# ─── 2. Revenue by category ──────────────────────────────────────────────────

def plot_revenue_by_category(df: pd.DataFrame) -> Path:
    revenue = (df.assign(revenue=df["quantity"] * df["item_price"])
               .groupby("product_category")["revenue"].sum()
               .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(revenue.index, revenue.values, color=sns.color_palette(PALETTE, len(revenue)))
    ax.set_title("Total Revenue by Product Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (₹)")
    ax.invert_yaxis()
    for bar, val in zip(bars, revenue.values):
        ax.text(val + revenue.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"₹{val:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, "02_revenue_by_category.png")


# ─── 3. Support distribution of frequent itemsets ────────────────────────────

def plot_support_distribution(frequent_itemsets: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(frequent_itemsets["support"], bins=20, color="#5A7FC1", edgecolor="white")
    axes[0].set_title("Support Distribution", fontweight="bold")
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("Frequency")

    length_counts = frequent_itemsets["length"].value_counts().sort_index()
    axes[1].bar(length_counts.index.astype(str), length_counts.values, color="#F4845F", edgecolor="white")
    axes[1].set_title("Itemset Size Distribution", fontweight="bold")
    axes[1].set_xlabel("Number of Items in Itemset")
    axes[1].set_ylabel("Count")

    fig.suptitle("Frequent Itemset Analysis", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "03_itemset_support_distribution.png")


# ─── 4. Top rules — scatter (support vs confidence, size=lift) ───────────────

def plot_rules_scatter(rules: pd.DataFrame, top_n: int = 50) -> Path:
    top = rules.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        top["support"], top["confidence"],
        c=top["lift"], s=top["lift"] * 80,
        cmap=PALETTE, alpha=0.7, edgecolors="grey", linewidths=0.5,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Lift", fontsize=10)
    ax.set_xlabel("Support", fontsize=11)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title(f"Top {top_n} Association Rules\n(bubble size ∝ lift)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_rules_scatter.png")


# ─── 5. Heatmap — top antecedents vs consequents by lift ─────────────────────

def plot_lift_heatmap(rules: pd.DataFrame, top_n: int = 15) -> Path:
    from association_rules import format_rules
    fmt = format_rules(rules.head(top_n * 5))
    pivot = fmt.pivot_table(
        index="antecedents", columns="consequents", values="lift", aggfunc="max"
    ).fillna(0)
    # Keep only top rows/cols by max lift
    row_max = pivot.max(axis=1).nlargest(top_n).index
    col_max = pivot.max(axis=0).nlargest(top_n).index
    pivot = pivot.loc[pivot.index.isin(row_max), pivot.columns.isin(col_max)]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9),
                                    max(5, len(pivot.index) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Lift"})
    ax.set_title("Lift Heatmap: Antecedents → Consequents", fontsize=13, fontweight="bold")
    ax.set_xlabel("Consequent")
    ax.set_ylabel("Antecedent")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return _save(fig, "05_lift_heatmap.png")


# ─── 6. Network graph of association rules ───────────────────────────────────

def plot_network_graph(rules: pd.DataFrame, top_n: int = 30) -> Path:
    from association_rules import format_rules
    fmt = format_rules(rules.head(top_n))

    G = nx.DiGraph()
    for _, row in fmt.iterrows():
        G.add_edge(row["antecedents"], row["consequents"],
                   weight=row["lift"], confidence=row["confidence"])

    lifts = [d["weight"] for _, _, d in G.edges(data=True)]
    max_lift = max(lifts) if lifts else 1

    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_sizes = [G.degree(n) * 300 + 500 for n in G.nodes()]
    edge_colors = [d["weight"] / max_lift for _, _, d in G.edges(data=True)]
    edge_widths = [1 + d["confidence"] * 4 for _, _, d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color="#5A7FC1", alpha=0.85, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths,
                                   edge_color=edge_colors, edge_cmap=plt.cm.YlOrRd,
                                   alpha=0.75, arrows=True,
                                   arrowstyle="-|>", arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white",
                            font_weight="bold", ax=ax)
    ax.set_title(f"Association Rules Network (Top {top_n} by Lift)\n"
                 "Node size ∝ degree  |  Edge width ∝ confidence  |  Edge colour ∝ lift",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(min(lifts), max_lift))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Lift", shrink=0.6)
    fig.tight_layout()
    return _save(fig, "06_network_graph.png")


# ─── 7. Reorder behaviour analysis ───────────────────────────────────────────

def plot_reorder_analysis(df: pd.DataFrame) -> Path:
    reorder = df.groupby("product_category")["is_reorder"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E07B54" if v > reorder.mean() else "#5A7FC1" for v in reorder]
    ax.bar(reorder.index, reorder.values * 100, color=colors)
    ax.axhline(reorder.mean() * 100, color="black", linestyle="--", linewidth=1.2, label=f"Mean {reorder.mean()*100:.1f}%")
    ax.set_title("Reorder Rate by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Reorder Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    high_patch = mpatches.Patch(color="#E07B54", label="Above average")
    low_patch  = mpatches.Patch(color="#5A7FC1", label="Below average")
    ax.legend(handles=[high_patch, low_patch], loc="upper right")
    fig.tight_layout()
    return _save(fig, "07_reorder_analysis.png")


# ─── 8. Monthly order trend ──────────────────────────────────────────────────

def plot_monthly_trend(df: pd.DataFrame) -> Path:
    df = df.copy()
    df["month_year"] = df["order_timestamp"].dt.to_period("M")
    monthly = df.groupby("month_year")["order_id"].nunique().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#5A7FC1", linewidth=2)
    ax.fill_between(monthly.index.astype(str), monthly.values, alpha=0.15, color="#5A7FC1")
    ax.set_title("Monthly Order Volume Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Orders")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save(fig, "08_monthly_trend.png")
