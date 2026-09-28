from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "analysis"

TRAJECTORIES_FILE = ANALYSIS_DIR / "post_change_learning_trajectories.csv"
SEED_METRICS_FILE = ANALYSIS_DIR / "seed_level_metrics.csv"
SUMMARY_FILE = ANALYSIS_DIR / "model_summary_metrics.csv"
LEARNING_FIGURE_FILE = ANALYSIS_DIR / "post_change_learning_curve.png"
RETENTION_FIGURE_FILE = ANALYSIS_DIR / "retention_curve.png"

DATA_SPLIT_SEED = 2026
PIXEL_PERMUTATION_SEED = 517
MODEL_SEEDS = [11, 22, 33, 44, 55]

PRE_CHANGE_EPOCHS = 20
POST_CHANGE_EPOCHS = 30
RECOVERY_THRESHOLD = 0.90

HIDDEN_UNITS = 32
LEARNING_RATE = 0.03
BATCH_SIZE = 64

MODEL_A_NAME = "Model A: L2 regularization"
MODEL_B_NAME = "Model B: no L2 regularization"

ANALYSIS_DIR.mkdir(exist_ok=True)
CLASSES = np.arange(10)


def make_model(seed, alpha):
    """Create one MLP with specified L2 regularization strength."""
    return MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation="relu",
        solver="sgd",
        learning_rate_init=LEARNING_RATE,
        momentum=0.0,
        alpha=alpha,
        batch_size=BATCH_SIZE,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=seed,
    )


def train_one_epoch(model, X_data, y_data, first_fit=False):
    """Train one MLP for one epoch."""
    if first_fit:
        model.partial_fit(X_data, y_data, classes=CLASSES)
    else:
        model.partial_fit(X_data, y_data)


def accuracy(model, X_data, y_data):
    """Calculate classification accuracy."""
    return accuracy_score(y_data, model.predict(X_data))


def first_epoch_at_threshold(values, threshold):
    """Return the first epoch reaching a specified performance threshold."""
    for epoch, value in enumerate(values, start=1):
        if value >= threshold:
            return epoch
    return np.nan


# -------------------------------------------------------------------
# 1. Load public data
# -------------------------------------------------------------------

digits = load_digits()

X = digits.data / 16.0
y = digits.target
indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=y,
    random_state=DATA_SPLIT_SEED,
)

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# -------------------------------------------------------------------
# 2. Create changed environment: fixed pixel permutation
# -------------------------------------------------------------------

permutation_rng = np.random.default_rng(PIXEL_PERMUTATION_SEED)
pixel_permutation = permutation_rng.permutation(X_train.shape[1])

X_train_changed = X_train[:, pixel_permutation]
X_test_changed = X_test[:, pixel_permutation]

# -------------------------------------------------------------------
# 3. Train models before and after environmental change
# -------------------------------------------------------------------

trajectory_rows = []
summary_rows = []

for seed in MODEL_SEEDS:
    model_a = make_model(seed=seed, alpha=0.001)
    model_b = make_model(seed=seed, alpha=0.0)

    # Phase 1: Original environment
    for epoch in range(PRE_CHANGE_EPOCHS):
        train_one_epoch(
            model_a,
            X_train,
            y_train,
            first_fit=(epoch == 0),
        )

        train_one_epoch(
            model_b,
            X_train,
            y_train,
            first_fit=(epoch == 0),
        )

    pre_change_a = accuracy(model_a, X_test, y_test)
    pre_change_b = accuracy(model_b, X_test, y_test)

    # Evaluate immediate effect of change before post-change training
    immediate_changed_a = accuracy(model_a, X_test_changed, y_test)
    immediate_changed_b = accuracy(model_b, X_test_changed, y_test)

    post_change_a = []
    post_change_b = []
    retention_a = []
    retention_b = []

    # Phase 2: Changed environment
    for post_change_epoch in range(1, POST_CHANGE_EPOCHS + 1):
        train_one_epoch(model_a, X_train_changed, y_train)
        train_one_epoch(model_b, X_train_changed, y_train)

        new_environment_a = accuracy(model_a, X_test_changed, y_test)
        new_environment_b = accuracy(model_b, X_test_changed, y_test)

        old_environment_a = accuracy(model_a, X_test, y_test)
        old_environment_b = accuracy(model_b, X_test, y_test)

        post_change_a.append(new_environment_a)
        post_change_b.append(new_environment_b)
        retention_a.append(old_environment_a)
        retention_b.append(old_environment_b)

        trajectory_rows.append(
            {
                "seed": seed,
                "model": MODEL_A_NAME,
                "post_change_epoch": post_change_epoch,
                "accuracy_new_environment": new_environment_a,
                "accuracy_old_environment": old_environment_a,
            }
        )

        trajectory_rows.append(
            {
                "seed": seed,
                "model": MODEL_B_NAME,
                "post_change_epoch": post_change_epoch,
                "accuracy_new_environment": new_environment_b,
                "accuracy_old_environment": old_environment_b,
            }
        )

    summary_rows.append(
        {
            "seed": seed,
            "model": MODEL_A_NAME,
            "l2_alpha": 0.001,
            "pre_change_accuracy": pre_change_a,
            "immediate_post_change_accuracy": immediate_changed_a,
            "immediate_decline": pre_change_a - immediate_changed_a,
            "accuracy_epoch_5": post_change_a[4],
            "accuracy_epoch_10": post_change_a[9],
            "final_new_accuracy": post_change_a[-1],
            "final_old_accuracy": retention_a[-1],
            "forgetting": pre_change_a - retention_a[-1],
            "epochs_to_90_percent": first_epoch_at_threshold(
                post_change_a,
                RECOVERY_THRESHOLD,
            ),
            "recovery_rate_first_10_epochs": (
                post_change_a[9] - immediate_changed_a
            ) / 10,
        }
    )

    summary_rows.append(
        {
            "seed": seed,
            "model": MODEL_B_NAME,
            "l2_alpha": 0.0,
            "pre_change_accuracy": pre_change_b,
            "immediate_post_change_accuracy": immediate_changed_b,
            "immediate_decline": pre_change_b - immediate_changed_b,
            "accuracy_epoch_5": post_change_b[4],
            "accuracy_epoch_10": post_change_b[9],
            "final_new_accuracy": post_change_b[-1],
            "final_old_accuracy": retention_b[-1],
            "forgetting": pre_change_b - retention_b[-1],
            "epochs_to_90_percent": first_epoch_at_threshold(
                post_change_b,
                RECOVERY_THRESHOLD,
            ),
            "recovery_rate_first_10_epochs": (
                post_change_b[9] - immediate_changed_b
            ) / 10,
        }
    )

# -------------------------------------------------------------------
# 4. Create and save result tables
# -------------------------------------------------------------------

trajectories = pd.DataFrame(trajectory_rows)
seed_metrics = pd.DataFrame(summary_rows)

summary_metrics = (
    seed_metrics.groupby("model")
    .agg(
        seeds=("seed", "count"),
        pre_change_accuracy_mean=("pre_change_accuracy", "mean"),
        pre_change_accuracy_sd=("pre_change_accuracy", "std"),
        immediate_post_change_accuracy_mean=(
            "immediate_post_change_accuracy",
            "mean",
        ),
        immediate_decline_mean=("immediate_decline", "mean"),
        accuracy_epoch_5_mean=("accuracy_epoch_5", "mean"),
        accuracy_epoch_10_mean=("accuracy_epoch_10", "mean"),
        recovery_rate_first_10_mean=(
            "recovery_rate_first_10_epochs",
            "mean",
        ),
        epochs_to_90_percent_mean=("epochs_to_90_percent", "mean"),
        final_new_accuracy_mean=("final_new_accuracy", "mean"),
        final_new_accuracy_sd=("final_new_accuracy", "std"),
        final_old_accuracy_mean=("final_old_accuracy", "mean"),
        forgetting_mean=("forgetting", "mean"),
    )
    .reset_index()
)

trajectories.to_csv(TRAJECTORIES_FILE, index=False)
seed_metrics.to_csv(SEED_METRICS_FILE, index=False)
summary_metrics.to_csv(SUMMARY_FILE, index=False)

# -------------------------------------------------------------------
# 5. Create and save post-change learning curve
# -------------------------------------------------------------------

curve_data = (
    trajectories.groupby(["model", "post_change_epoch"])[
        "accuracy_new_environment"
    ]
    .agg(["mean", "std", "count"])
    .reset_index()
)

curve_data["sem"] = curve_data["std"] / np.sqrt(curve_data["count"])

colors = {
    MODEL_A_NAME: "#4C78A8",
    MODEL_B_NAME: "#F58518",
}

fig, ax = plt.subplots(figsize=(10, 6))

for model_name, model_data in curve_data.groupby("model"):
    ax.plot(
        model_data["post_change_epoch"],
        model_data["mean"],
        label=model_name,
        color=colors[model_name],
        linewidth=2,
    )

    ax.fill_between(
        model_data["post_change_epoch"],
        model_data["mean"] - model_data["sem"],
        model_data["mean"] + model_data["sem"],
        color=colors[model_name],
        alpha=0.20,
    )

ax.axhline(
    RECOVERY_THRESHOLD,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="90% recovery threshold",
)

ax.set_title("Post-Change Learning Curves")
ax.set_xlabel("Epochs After Environmental Change")
ax.set_ylabel("Accuracy on Changed Environment")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(LEARNING_FIGURE_FILE, dpi=300)
plt.close()

# -------------------------------------------------------------------
# 6. Create and save retention curve
# -------------------------------------------------------------------

retention_data = (
    trajectories.groupby(["model", "post_change_epoch"])[
        "accuracy_old_environment"
    ]
    .agg(["mean", "std", "count"])
    .reset_index()
)

retention_data["sem"] = (
    retention_data["std"] / np.sqrt(retention_data["count"])
)

fig, ax = plt.subplots(figsize=(10, 6))

for model_name, model_data in retention_data.groupby("model"):
    ax.plot(
        model_data["post_change_epoch"],
        model_data["mean"],
        label=model_name,
        color=colors[model_name],
        linewidth=2,
    )

    ax.fill_between(
        model_data["post_change_epoch"],
        model_data["mean"] - model_data["sem"],
        model_data["mean"] + model_data["sem"],
        color=colors[model_name],
        alpha=0.20,
    )

ax.set_title("Retention of Original Environment")
ax.set_xlabel("Epochs After Environmental Change")
ax.set_ylabel("Accuracy on Original Environment")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(RETENTION_FIGURE_FILE, dpi=300)
plt.close()

# -------------------------------------------------------------------
# 7. Print reproducibility and results summary
# -------------------------------------------------------------------

print("\nReplication Lab 2: Environmental Change Experiment")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"Digits observations: {len(X):,}")
print(f"Training observations: {len(X_train):,}")
print(f"Test observations: {len(X_test):,}")
print(f"Model seeds: {MODEL_SEEDS}")
print(f"Pre-change epochs: {PRE_CHANGE_EPOCHS}")
print(f"Post-change epochs: {POST_CHANGE_EPOCHS}")
print(f"Recovery threshold: {RECOVERY_THRESHOLD:.0%}")

print("\nModel summary metrics")
print(summary_metrics.round(4).to_string(index=False))

print(f"\nSaved trajectories: {TRAJECTORIES_FILE}")
print(f"Saved seed metrics: {SEED_METRICS_FILE}")
print(f"Saved summary table: {SUMMARY_FILE}")
print(f"Saved learning curve: {LEARNING_FIGURE_FILE}")
print(f"Saved retention curve: {RETENTION_FIGURE_FILE}")
