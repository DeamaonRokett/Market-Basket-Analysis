# 🛒 Market Basket Analysis

A complete, end-to-end **Market Basket Analysis** pipeline built with Python.  
It mines association rules from e-commerce transaction data using the **Apriori algorithm** and powers a lightweight **recommendation engine**.

---

##  Project Structure

```
market_basket_analysis/
│
├── data/
│   └── generated_data.csv        # Raw transaction data
│
├── src/
│   ├── preprocess.py             # Data loading, EDA, basket matrix construction
│   ├── association_rules.py      # Apriori mining, rule filtering & formatting
│   ├── visualise.py              # All 8 chart generators
│   └── recommender.py            # Rule-based recommendation engine
│
├── outputs/                      # Auto-generated charts + rules CSV
├── tests/
│   └── test_pipeline.py          # 20+ unit tests (pytest)
│
├── main.py                       # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

##  Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/market-basket-analysis.git
cd market-basket-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Run with custom thresholds
python main.py --min-support 0.03 --min-lift 1.2 --min-confidence 0.4

# 5. Product-level analysis (instead of category-level)
python main.py --level product_name --min-support 0.01

# 6. Run tests
pytest tests/ -v
```

---

##  What the Pipeline Does

### Step 1 — Data Loading & EDA
- Loads CSV with 1,000 transactions across 22 product categories
- Reports order/user/product counts and date range
- Adds temporal features (hour, day of week, month, is_weekend)

### Step 2 — Basket Matrix Construction
- Pivots transactions into a binary **order × category** matrix
- Supports both **category-level** and **product-level** granularity

### Step 3 — Apriori Mining
- Mines **frequent itemsets** via `mlxtend`'s Apriori implementation
- Generates association rules filtered by **support**, **confidence**, and **lift**
- Saves top rules to `outputs/association_rules.csv`

### Step 4 — Visualisations (8 charts)

| Chart | Description |
|-------|-------------|
| `01_category_distribution.png` | Transaction count per category |
| `02_revenue_by_category.png` | Total revenue breakdown |
| `03_itemset_support_distribution.png` | Support histogram + itemset sizes |
| `04_rules_scatter.png` | Support vs Confidence scatter (bubble = lift) |
| `05_lift_heatmap.png` | Antecedent → Consequent lift heatmap |
| `06_network_graph.png` | Rule network (node = item, edge = rule) |
| `07_reorder_analysis.png` | Reorder rate by category |
| `08_monthly_trend.png` | Monthly order volume trend |

### Step 5 — Recommendation Engine
- `BasketRecommender` class takes existing basket items and recommends new ones
- Ranked by **lift**, deduplicated, configurable top-K
- Supports batch recommendations for multiple users simultaneously

---

##  CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/generated_data.csv` | Path to raw CSV |
| `--level` | `category` | Basket level: `category` or `product_name` |
| `--min-support` | `0.05` | Minimum support threshold |
| `--min-lift` | `1.0` | Minimum lift threshold |
| `--min-confidence` | `0.3` | Minimum confidence threshold |
| `--top-k-recs` | `5` | Recommendations to return per basket |
| `--skip-plots` | `False` | Skip chart generation |

---

##  Key Concepts

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Support** | P(A ∩ B) | How often A and B appear together |
| **Confidence** | P(B \| A) = P(A ∩ B) / P(A) | When A is bought, how often is B also bought |
| **Lift** | Confidence / P(B) | How much more likely B is given A vs randomly |

- **Lift > 1** → A and B are positively associated (co-purchase more than chance)  
- **Lift = 1** → statistically independent  
- **Lift < 1** → negative association

---

##  Tests

```bash
pytest tests/test_pipeline.py -v
```

Covers:
- Data loading and schema validation
- Basket matrix shape and binary encoding
- Apriori output bounds (support ∈ [0,1])
- Rule filtering and formatting
- Recommender top-K, deduplication, edge cases

---

##  Dependencies

```
pandas, numpy, matplotlib, seaborn
mlxtend          # Apriori & association rules
networkx         # Network graph
scikit-learn     # Preprocessing utilities
pytest           # Testing
```

---

##  Extending This Project

- **Swap the algorithm**: Replace Apriori with FP-Growth (`mlxtend.frequent_patterns.fpgrowth`) for faster mining on large datasets
- **Add a Streamlit dashboard**: Wire `visualise.py` charts into a `streamlit run app.py` UI
- **Segment by time**: Filter basket matrix to weekends/evenings for temporal rule mining
- **Personalise**: Join with user demographic data for cohort-specific rule sets

---

## 📄 License

MIT — free to use, modify, and distribute.
