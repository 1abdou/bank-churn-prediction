import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


DATA_PATH = "customer_churn.csv"
OUTPUT_DIR = "plots"
TARGET = "churn"
sns.set_theme(style="whitegrid")

def plot_annotated_boxplot(series: pd.Series, title: str) -> None:
    """Box plot with the median, mean and outlier count highlighted."""
    series = series.dropna()
    median, mean = series.median(), series.mean()
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    n_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

    fig, ax = plt.subplots(figsize=(5, 6))
    sns.boxplot(
        y=series,
        ax=ax,
        color="lightsteelblue",
        showmeans=True,
        medianprops={"color": "crimson", "linewidth": 2.5},
        meanprops={"marker": "D", "markerfacecolor": "gold",
                   "markeredgecolor": "black", "markersize": 9},
        flierprops={"marker": "o", "markerfacecolor": "gray", "alpha": 0.5},
    )

    legend_items = [
        Line2D([], [], color="crimson", linewidth=2.5, label=f"Median: {median:.2f}"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="gold",
               markeredgecolor="black", markersize=9, label=f"Mean: {mean:.2f}"),
        Line2D([], [], linestyle="", label=f"Q1: {q1:.2f}  |  Q3: {q3:.2f}"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="gray",
               alpha=0.5, label=f"Outliers: {n_outliers}"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title(title)
    fig.tight_layout()

# 1. Load the dataset
df = pd.read_csv(DATA_PATH)

# 2. Lowercase the column headings
df.columns = df.columns.str.lower()

# 3. Dataset shape
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

# 4. Data types of features
print("Data types:")
print(df.dtypes, "\n")

# 5. Missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# 6. Duplicates
print(f"Duplicate rows: {df.duplicated().sum()}\n")

# 7. Box plot for each (numeric) feature, saved separately in box_plots/
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    plot_annotated_boxplot(df[col], title=f"Box Plot: {col}")
    plt.savefig(f"{OUTPUT_DIR}/box plots/{col}_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"Saved {len(numeric_cols)} box plots to: {OUTPUT_DIR}/box plots\n")

# 8. Correlation heatmap (lower triangle only)
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap (Lower Triangle)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/corr_map.png")


# 9. Label distribution
counts = df[TARGET].value_counts()
print("Label distribution:")
print(counts, "\n")

plt.figure(figsize=(6, 4))
ax = sns.barplot(x=counts.index.astype(str), y=counts.values)
ax.bar_label(ax.containers[0])
plt.title("Churn Distribution")
plt.xlabel(TARGET)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/targit_dist.png")
