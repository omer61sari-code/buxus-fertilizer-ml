# -*- coding: utf-8 -*-
"""
FINAL REVISED MACHINE-LEARNING ANALYSIS - PYTHON ONLY

Design
------
70 observations: 7 treatment classes x 10 observations
Stratified 70:30 split: 49 training + 21 untouched hold-out
5-fold CV is performed only on the 49-observation training set.
Algorithms retained from the manuscript:
    IBk, PART, J48, Multilayer Perceptron, Naive Bayes
Random Forest is used only for feature importance and SHAP explainability.

IMPORTANT
---------
This final version uses Python only; Weka/Java are not required.
IBk       -> sklearn KNeighborsClassifier
J48       -> sklearn DecisionTreeClassifier (C4.5-style tree cannot be exactly
             reproduced by sklearn; this is a reproducible Python decision-tree
             implementation for the manuscript comparison.)
MLP       -> sklearn MLPClassifier
Naive Bayes -> sklearn GaussianNB
PART      -> a Python sequential rule learner based on pruned decision-tree
             rules. It is intentionally kept separate from J48.

If the manuscript specifically states "Weka PART" or "Weka J48", that wording
should be changed unless the exact Weka implementation is used. Do not claim
software-specific identity that was not used in the final analysis.
"""

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_FOLDS = 5

BASE_FOLDER = Path(__file__).resolve().parent
RESULT_FOLDER = BASE_FOLDER / "Results_ML_Revised"
FIGURE_FOLDER = BASE_FOLDER / "Figures_ML_Revised"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = BASE_FOLDER / "input" / "veriler.xlsx"

# Prefer the validated 70-observation CSV if it exists.
CSV_CANDIDATES = [
    BASE_FOLDER / "ML_70_current_input.csv",
    Path.home() / "Downloads" / "ML_70_current_input.csv",
    Path.home() / "Desktop" / "ML_70_current_input.csv",
]
DATA_PATH = next((p for p in CSV_CANDIDATES if p.exists()), None)

TARGET = "Applications"
FEATURES = [
    "Plant_height",
    "Plant_width",
    "Number_of_shoots",
    "Shoot_length",
    "Leaf_width",
    "Leaf_length",
]

CLASS_ORDER = [
    "Control",
    "Potassium",
    "Nitrogen",
    "Vermicompost",
    "Bacteria",
    "Mycorrhizal",
    "Sheep wool",
]

# ============================================================
# COLUMN NORMALIZATION
# ============================================================
def norm_col(x):
    s = str(x).strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s

ALIASES = {
    "Applications": ["Applications", "Application", "applications", "Treatment", "Treatments"],
    "Plant_height": ["Plant_height", "Plant height", "Plant Height"],
    "Plant_width": ["Plant_width", "Plant width", "Plant Width"],
    "Number_of_shoots": ["Number_of_shoots", "Number of shoots", "Number of shoots"],
    "Shoot_length": ["Shoot_length", "Shoot length", "Shoot Length"],
    "Leaf_width": ["Leaf_width", "Leaf width", "Leaf Width"],
    "Leaf_length": ["Leaf_length", "Leaf length", "Leaf Length"],
}


def standardize_columns(df):
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]
    rename = {}
    for canonical, aliases in ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename[a] = canonical
                break
    df = df.rename(columns=rename)
    return df


# ============================================================
# READ DATA
# ============================================================
if DATA_PATH is not None:
    print(f"Using validated ML CSV: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
else:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            "ML_70_current_input.csv bulunamadı ve Excel dosyası da bulunamadı.\n"
            f"Excel yolu: {EXCEL_PATH}"
        )
    print(f"Using Excel input: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

df = standardize_columns(df)

required = [TARGET] + FEATURES
missing = [c for c in required if c not in df.columns]
if missing:
    print("Available columns:")
    print(list(df.columns))
    raise ValueError(f"Missing required columns: {missing}")

# Numeric conversion only for the six predictors.
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[TARGET] = df[TARGET].astype(str).str.strip()
df = df.dropna(subset=required).copy()

# Keep exactly the seven manuscript classes.
df = df[df[TARGET].isin(CLASS_ORDER)].copy()

print("\nDATA CHECK")
print("Rows:", len(df))
print("Class counts:")
print(df[TARGET].value_counts().reindex(CLASS_ORDER).to_string())

if len(df) != 70:
    raise ValueError(
        f"Expected the manuscript ML dataset to contain 70 observations; found {len(df)}."
    )

counts = df[TARGET].value_counts()
if any(counts.get(c, 0) != 10 for c in CLASS_ORDER):
    raise ValueError("Expected exactly 10 observations in each of the seven treatment classes.")


# ============================================================
# CUSTOM PYTHON PART-LIKE RULE LEARNER
# ============================================================
class PARTClassifier(BaseEstimator, ClassifierMixin):
    """Simple sequential covering rule learner using pruned tree leaves.

    This is a Python rule learner inspired by PART's partial-tree/rule idea.
    It is not a byte-for-byte reproduction of Weka's PART implementation.
    """
    def __init__(self, max_depth=3, min_samples_leaf=2, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.global_class_ = pd.Series(y).mode().iloc[0]
        self.rules_ = []
        remaining = np.arange(len(y))
        max_rules = max(5, len(self.classes_) * 2)

        for _ in range(max_rules):
            if len(remaining) < self.min_samples_leaf:
                break
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                criterion="gini",
            )
            tree.fit(X[remaining], y[remaining])
            leaf_id = tree.apply(X[remaining])
            leaves = np.unique(leaf_id)
            best = None
            for leaf in leaves:
                idx_local = np.where(leaf_id == leaf)[0]
                idx_global = remaining[idx_local]
                yy = y[idx_global]
                counts = pd.Series(yy).value_counts()
                pred = counts.index[0]
                purity = counts.iloc[0] / len(yy)
                score = purity * np.sqrt(len(yy))
                if best is None or score > best[0]:
                    best = (score, leaf, idx_global, pred, purity)
            if best is None:
                break
            _, leaf, idx_global, pred, purity = best
            self.rules_.append((tree, int(leaf), pred, float(purity)))
            remaining = np.array([i for i in remaining if i not in set(idx_global)])
            if len(remaining) == 0:
                break

        self._fallback_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.global_class_, dtype=object)
        assigned = np.zeros(X.shape[0], dtype=bool)
        for tree, leaf, rule_class, _purity in self.rules_:
            ids = tree.apply(X)
            mask = (ids == leaf) & (~assigned)
            pred[mask] = rule_class
            assigned[mask] = True
        if (~assigned).any():
            pred[~assigned] = self._fallback_tree.predict(X[~assigned])
        return pred

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        proba = np.zeros(
            (X.shape[0], len(self.classes_)),
            dtype=float
        )
        class_to_index = {
            cls: i for i, cls in enumerate(self.classes_)
        }
        assigned = np.zeros(X.shape[0], dtype=bool)

        for tree, leaf, rule_class, purity in self.rules_:
            leaf_ids = tree.apply(X)
            mask = (leaf_ids == leaf) & (~assigned)

            if mask.any():
                j = class_to_index.get(rule_class)
                if j is not None:
                    proba[mask, j] = purity
                    if len(self.classes_) > 1 and purity < 1:
                        residual = (
                            (1.0 - purity) /
                            (len(self.classes_) - 1)
                        )
                        for jj in range(len(self.classes_)):
                            if jj != j:
                                proba[mask, jj] = residual

                assigned[mask] = True

        if (~assigned).any():
            idx = np.where(~assigned)[0]
            fallback_prob = self._fallback_tree.predict_proba(X[idx])

            for j, cls in enumerate(self._fallback_tree.classes_):
                if cls in class_to_index:
                    proba[
                        idx,
                        class_to_index[cls]
                    ] = fallback_prob[:, j]

        row_sums = proba.sum(axis=1, keepdims=True)
        proba = np.divide(
            proba,
            row_sums,
            out=np.full_like(
                proba,
                1.0 / len(self.classes_)
            ),
            where=row_sums != 0
        )

        return proba


# ============================================================
# ADDITIONAL PERFORMANCE METRICS
# ============================================================

def get_probability_predictions(model, X_data, class_order):
    """Return probabilities aligned to class_order when available."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = np.asarray(model.predict_proba(X_data))
        model_classes = getattr(model, "classes_", None)

        if model_classes is None and hasattr(model, "named_steps"):
            final_model = model.named_steps.get("model")
            if final_model is not None:
                model_classes = getattr(final_model, "classes_", None)

        if model_classes is None:
            return None

        aligned = np.zeros(
            (len(X_data), len(class_order)),
            dtype=float
        )

        for j, cls in enumerate(model_classes):
            if cls in class_order:
                aligned[:, class_order.index(cls)] = probabilities[:, j]

        row_sums = aligned.sum(axis=1, keepdims=True)
        aligned = np.divide(
            aligned,
            row_sums,
            out=np.zeros_like(aligned),
            where=row_sums != 0
        )

        return aligned

    except Exception:
        return None


def multiclass_auc(model, X_data, y_true, class_order):
    """Weighted one-vs-rest ROC-AUC where probability outputs are available."""
    probabilities = get_probability_predictions(
        model, X_data, class_order
    )

    if probabilities is None:
        return np.nan

    try:
        y_onehot = np.zeros(
            (len(y_true), len(class_order)),
            dtype=int
        )

        class_to_index = {
            cls: i for i, cls in enumerate(class_order)
        }

        for i, cls in enumerate(y_true):
            if cls in class_to_index:
                y_onehot[i, class_to_index[cls]] = 1

        represented = y_onehot.sum(axis=0) > 0

        if represented.sum() < 2:
            return np.nan

        return float(
            roc_auc_score(
                y_onehot[:, represented],
                probabilities[:, represented],
                average="weighted",
                multi_class="ovr"
            )
        )

    except Exception:
        return np.nan


# ============================================================
# MODELS - ONLY THE FIVE MANUSCRIPT ALGORITHMS
# ============================================================
models = {
    "IBk": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=1)),
    ]),
    "PART": PARTClassifier(max_depth=3, min_samples_leaf=2, random_state=RANDOM_STATE),
    "J48": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    ),
    "Multilayer Perceptron": Pipeline([
        ("scale", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(10,),
            activation="relu",
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ]),
    "Naive Bayes": GaussianNB(),
}

# ============================================================
# STRATIFIED 70:30 SPLIT
# ============================================================
X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining observations: {len(X_train)}")
print(f"Untouched hold-out observations: {len(X_holdout)}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results = []
holdout_predictions = {}
confusions = {}

for name, model in models.items():
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    # 1. 5-fold CV on training only.
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
    cv_accuracy = float(np.mean(cv_scores))
    cv_balanced = float(np.mean(cross_val_score(
        model, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=None
    )))

    # 2. Fit once on all 49 training observations.
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_balanced = balanced_accuracy_score(y_train, train_pred)

    # 3. One final prediction on untouched 21 observations.
    hold_pred = model.predict(X_holdout)
    hold_accuracy = accuracy_score(y_holdout, hold_pred)
    hold_balanced = balanced_accuracy_score(y_holdout, hold_pred)

    labels = CLASS_ORDER
    cm = confusion_matrix(y_holdout, hold_pred, labels=labels)
    confusions[name] = cm
    holdout_predictions[name] = hold_pred

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_holdout,
        hold_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    hold_kappa = cohen_kappa_score(y_holdout, hold_pred)
    hold_auc = multiclass_auc(
        model,
        X_holdout,
        y_holdout,
        labels
    )

    results.append({
        "Model": name,
        "Training Accuracy": train_accuracy,
        "5-fold CV Accuracy": cv_accuracy,
        "5-fold CV Balanced Accuracy": cv_balanced,
        "Hold-out Accuracy": hold_accuracy,
        "Hold-out Balanced Accuracy": hold_balanced,
        "Hold-out Precision": precision,
        "Hold-out Recall": recall,
        "Hold-out F1": f1,
        "Hold-out Kappa": hold_kappa,
        "Hold-out AUC": hold_auc,
    })

    print(f"Training accuracy       : {train_accuracy:.4f}")
    print(f"5-fold CV accuracy      : {cv_accuracy:.4f}")
    print(f"5-fold CV balanced acc. : {cv_balanced:.4f}")
    print(f"Hold-out accuracy       : {hold_accuracy:.4f}")
    print(f"Hold-out balanced acc.  : {hold_balanced:.4f}")
    print(f"Hold-out kappa          : {hold_kappa:.4f}")
    print(
        "Hold-out ROC-AUC        : "
        + (
            "N/A"
            if np.isnan(hold_auc)
            else f"{hold_auc:.4f}"
        )
    )

results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_FOLDER / "Model_Performance_Python.csv", index=False)

table_cols = [
    "Model",
    "Training Accuracy",
    "5-fold CV Accuracy",
    "5-fold CV Balanced Accuracy",
    "Hold-out Accuracy",
    "Hold-out Balanced Accuracy",
    "Hold-out Precision",
    "Hold-out Recall",
    "Hold-out F1",
    "Hold-out Kappa",
    "Hold-out AUC",
]

results_table = results_df[table_cols].copy()

results_table.to_csv(
    RESULT_FOLDER / "Table4_ML_Classification_Performance.csv",
    index=False
)

results_table.to_excel(
    RESULT_FOLDER / "Table4_ML_Classification_Performance.xlsx",
    index=False
)

summary_lines = []
for _, r in results_df.iterrows():
    summary_lines.append(
        f"{r['Model']}: Training accuracy={r['Training Accuracy']:.4f}; "
        f"5-fold CV accuracy={r['5-fold CV Accuracy']:.4f}; "
        f"Hold-out accuracy={r['Hold-out Accuracy']:.4f}; "
        f"Hold-out balanced accuracy={r['Hold-out Balanced Accuracy']:.4f}; "
        f"Hold-out F1={r['Hold-out F1']:.4f}"
    )
(RESULT_FOLDER / "Manuscript_ML_Summary.txt").write_text(
    "\n".join(summary_lines), encoding="utf-8"
)

print("\n" + "=" * 72)
print("FINAL PYTHON METRICS")
print("=" * 72)
print(results_df.to_string(index=False))

per_class_rows = []
for name, pred in holdout_predictions.items():
    p, r, f, s = precision_recall_fscore_support(
        y_holdout, pred, labels=CLASS_ORDER, zero_division=0
    )
    for cls, pp, rr, ff, ss in zip(CLASS_ORDER, p, r, f, s):
        per_class_rows.append({
            "Model": name,
            "Class": cls,
            "Precision": pp,
            "Recall": rr,
            "F1": ff,
            "Support": ss
        })
pd.DataFrame(per_class_rows).to_csv(
    RESULT_FOLDER / "Holdout_PerClass_Metrics.csv", index=False
)

# ============================================================
# FIGURE 1 - MODEL PERFORMANCE
# ============================================================
plot_df = results_df.copy()
metric_cols = ["Training Accuracy", "5-fold CV Accuracy", "Hold-out Balanced Accuracy"]
labels = plot_df["Model"].tolist()
xpos = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 7))
for i, col in enumerate(metric_cols):
    ax.bar(xpos + (i - 1) * width, plot_df[col], width, label=col)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=12, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy / balanced accuracy")
ax.set_title("Comparison of Machine Learning Classification Performance")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURE_FOLDER / "Figure10_ModelComparison_Python.png", dpi=600, bbox_inches="tight")
plt.savefig(FIGURE_FOLDER / "Figure10_ModelComparison_Python.pdf", dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 2 - NAIVE BAYES HOLD-OUT CONFUSION MATRIX
# ============================================================
def plot_cm(cm, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=12)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / filename, dpi=600, bbox_inches="tight")
    plt.show()

plot_cm(
    confusions["Naive Bayes"],
    "Confusion Matrix - Naive Bayes (Hold-out set)",
    "Figure11_NaiveBayes_ConfusionMatrix.png",
)

# ============================================================
# RANDOM FOREST - EXPLAINABILITY ONLY
# ============================================================
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight=None,
)
rf.fit(X_train, y_train)
rf_hold_pred = rf.predict(X_holdout)

# Feature importance.
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
importance.to_csv(RESULT_FOLDER / "RandomForest_FeatureImportance.csv", header=["Importance"])

fig, ax = plt.subplots(figsize=(10, 7))
importance.plot(kind="barh", ax=ax)
ax.set_xlabel("Relative importance")
ax.set_title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(FIGURE_FOLDER / "Figure12_RandomForest_FeatureImportance.png", dpi=600, bbox_inches="tight")
plt.savefig(FIGURE_FOLDER / "Figure12_RandomForest_FeatureImportance.pdf", dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# SHAP - ROBUST MULTICLASS HANDLING
# ============================================================
try:
    import shap

    Xh = X_holdout.copy()
    explainer = shap.TreeExplainer(rf)
    shap_result = explainer(Xh)

    values = shap_result.values if hasattr(shap_result, "values") else shap_result
    values = np.asarray(values)

    # Modern SHAP: (n_samples, n_features, n_classes)
    if values.ndim == 3:
        predicted = rf_hold_pred
        class_to_idx = {c: i for i, c in enumerate(rf.classes_)}
        selected = np.empty((len(Xh), len(FEATURES)), dtype=float)
        for i, cls in enumerate(predicted):
            selected[i] = values[i, :, class_to_idx[cls]]
        shap_selected = selected
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 2:
        shap_selected = values
        mean_abs = np.mean(np.abs(values), axis=0)
    elif isinstance(shap_result, list):
        arr = np.asarray(shap_result)
        if arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))
            shap_selected = np.mean(arr, axis=0)
        else:
            raise ValueError(f"Unexpected SHAP list shape: {arr.shape}")
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    # Publication-safe guard against the old 1e-17 failure.
    if not np.isfinite(shap_selected).all():
        raise ValueError("SHAP contains non-finite values.")
    if np.nanmax(np.abs(shap_selected)) < 1e-10:
        raise ValueError(
            "SHAP values are effectively zero. This indicates a model/explainer "
            "configuration problem, not a meaningful feature effect."
        )

    mean_abs_s = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=True)
    mean_abs_s.to_csv(RESULT_FOLDER / "RandomForest_MeanAbsolute_SHAP.csv", header=["MeanAbsSHAP"])

    # SHAP bar plot.
    fig, ax = plt.subplots(figsize=(10, 7))
    mean_abs_s.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Mean Absolute SHAP Importance - Hold-out Observations")
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / "Figure13_RandomForest_MeanAbsSHAP.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIGURE_FOLDER / "Figure13_RandomForest_MeanAbsSHAP.pdf", dpi=600, bbox_inches="tight")
    plt.show()

    # SHAP summary using predicted-class contribution per hold-out observation.
    shap.summary_plot(
        shap_selected,
        Xh,
        feature_names=FEATURES,
        show=False,
        plot_size=(11, 7),
    )
    plt.title("SHAP Summary Plot - Random Forest Hold-out Observations")
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / "Figure14_RandomForest_SHAP_Summary.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIGURE_FOLDER / "Figure14_RandomForest_SHAP_Summary.pdf", dpi=600, bbox_inches="tight")
    plt.show()

except Exception as exc:
    print("\nSHAP analysis could not be completed safely:")
    print(type(exc).__name__, str(exc))
    print("No misleading SHAP figure was produced.")

# ============================================================
# SAVE HOLD-OUT PREDICTIONS
# ============================================================
holdout_out = X_holdout.copy()
holdout_out[TARGET] = y_holdout.values
for name, pred in holdout_predictions.items():
    holdout_out[f"Pred_{name}"] = pred
holdout_out.to_csv(RESULT_FOLDER / "Holdout_Predictions_All_Models.csv", index=False)

print("\nANALYSIS COMPLETED - PYTHON ONLY")
print(f"Results folder : {RESULT_FOLDER}")
print(f"Figures folder : {FIGURE_FOLDER}")
