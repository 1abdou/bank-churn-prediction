from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

VARIANCE_THRESHOLD = 0.95  # keep enough components to explain 95% of the variance
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent  # same folder as this script
DATA_DIR = BASE_DIR / "processed_data"
PLOTS_DIR = BASE_DIR / "plots"

sns.set_theme(style="whitegrid")


def save_figure(filename: str) -> None:
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")


# 1. Load the preprocessed (split and scaled) data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

# 2. Fit PCA with all components on the training set to inspect the variance
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_components = int(np.argmax(cumulative >= VARIANCE_THRESHOLD)) + 1

variance_table = pd.DataFrame(
    {
        "explained_variance": explained.round(4),
        "cumulative_variance": cumulative.round(4),
    },
    index=[f"PC{i}" for i in range(1, len(explained) + 1)],
)
print("Explained variance per component:")
print(variance_table, "\n")
print(f"Components needed for {VARIANCE_THRESHOLD:.0%} variance: {n_components}\n")

# 3. Scree plot (individual and cumulative explained variance)
components_axis = np.arange(1, len(explained) + 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(components_axis, explained, alpha=0.7, label="Individual")
ax.plot(components_axis, cumulative, marker="o", color="crimson", label="Cumulative")
ax.axhline(VARIANCE_THRESHOLD, color="gray", linestyle="--", label=f"{VARIANCE_THRESHOLD:.0%} threshold")
ax.axvline(n_components, color="gray", linestyle=":")
ax.set_xticks(components_axis)
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("PCA Scree Plot")
ax.legend(loc="center right")
plt.tight_layout()
save_figure("pca_scree_plot.png")
plt.show()

# 4. Final PCA: fit on train only, then transform both sets
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
pc_columns = [f"pc{i}" for i in range(1, n_components + 1)]

X_train_pca = pd.DataFrame(pca.fit_transform(X_train), columns=pc_columns)
X_test_pca = pd.DataFrame(pca.transform(X_test), columns=pc_columns)
print(f"Reduced train shape: {X_train_pca.shape} | Reduced test shape: {X_test_pca.shape}\n")

# 5. Component loadings: how each original feature contributes to each component
loadings = pd.DataFrame(pca.components_.T, index=X_train.columns, columns=pc_columns)

plt.figure(figsize=(1.1 * n_components + 3, 6))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("PCA Component Loadings")
plt.tight_layout()
save_figure("pca_loadings_heatmap.png")
plt.show()

# 6. 2D projection on the first two components, colored by churn
plt.figure(figsize=(7, 6))
sns.scatterplot(
    x=X_train_pca["pc1"], y=X_train_pca["pc2"], hue=y_train, palette="Set1", alpha=0.5, s=20
)
plt.xlabel(f"PC1 ({explained[0]:.1%} variance)")
plt.ylabel(f"PC2 ({explained[1]:.1%} variance)")
plt.title("Training Data Projected on the First Two Components")
plt.legend(title="churn")
plt.tight_layout()
save_figure("pca_2d_projection.png")
plt.show()

# 7. Save the PCA-transformed data and the fitted PCA
X_train_pca.to_csv(DATA_DIR / "X_train_pca.csv", index=False)
X_test_pca.to_csv(DATA_DIR / "X_test_pca.csv", index=False)
joblib.dump(pca, DATA_DIR / "pca.joblib")

print(f"Saved PCA data and model to: {DATA_DIR}")
print(f"Saved plots to: {PLOTS_DIR}")