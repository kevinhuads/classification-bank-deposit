# Core numerical / data-manipulation libraries
import numpy as np     
import pandas as pd   
import shap

import os, requests
from dotenv import load_dotenv

# Statistical test used inside cramers_v
from scipy.stats import chi2_contingency

# Plotting the LightGBM learning curve
import matplotlib.pyplot as plt        
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import seaborn as sns

from matplotlib.ticker import MultipleLocator, PercentFormatter
from matplotlib.lines import Line2D
    
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

N_FOLDS = 5
SEED = 3

# Seaborn custom theme
sns.set_theme(
    style="darkgrid",           
    rc={
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor":   "#0d1b2a",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#2a3f5f",
        "axes.labelcolor":  "#ffffff",
        "text.color":       "#ffffff",
        "xtick.color":      "#ffffff",
        "ytick.color":      "#ffffff",
    },
    palette="deep"               
)

# --------------------------- EDA Notebook ------------------------------

# Customize tables ------

dtype_palette = {
    "int64"         : "blue",
    "float64"       : "green",
    "object"        : "darkorange",
    "bool"          : "purple",
    "category"      : "teal",
    "datetime64[ns]": "brown"
}

# --- make all text white ---
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

def colour_dtype(val):
    """Return a CSS 'color' style based on a dtype name."""
    return f"color: {dtype_palette.get(str(val), 'black')};"

# Value-based gradient for the missing column
def colour_gradient(val, cmap_name, vmin, vmax):
    """Return a CSS 'color' style using a colormap scaled to [vmin, vmax]."""
    if pd.isna(val):
        return ""
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgb   = plt.get_cmap(cmap_name)(norm(val))[:3]        
    r,g,b = (int(255*c) for c in rgb)
    return f"color: rgb({r}, {g}, {b});"
    
# Function to calculate Cramér's V ---------
def cramers_v(x, y):
    """Compute Cramér's V for association between two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))
    
# Compute the Eta squared after the Anova test --------
def compute_eta_squared(df, x, y):
    """Compute eta-squared (effect size) from one-way ANOVA for x→y."""
    groups = [g[y].values for _, g in df[[x, y]].dropna().groupby(x)]
    grand_mean = df[y].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df[y] - grand_mean)**2)
    return ss_between / ss_total if ss_total != 0 else float('nan')
    
# Human-friendly axis formatter (K/M) and small ylim padding for internal margin
def human_fmt(x, pos):
    if x >= 1_000_000:
        s = f"{x/1_000_000:.1f}M"
    elif x >= 1_000:
        s = f"{x/1_000:.1f}K"
    else:
        s = f"{int(x):,}"
    return s.replace('.0M', 'M').replace('.0K', 'K')


def plot_feature(df, col, y, i, log_scale=False, remove_outliers=False):
    k = 2*i
    x = df[col]

    if remove_outliers:
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            x = x[(x >= lo) & (x <= hi)]
    if log_scale:
        x = x[x > 0]

    y_sub = y.loc[x.index]
    # -----------------------------------------------------------------------
    # left-hand histogram
    axes[k].hist(x[y_sub == 0], bins=60, density=True, alpha=0.55, label="y = 0")
    axes[k].hist(x[y_sub == 1], bins=60, density=True, alpha=0.55, label="y = 1")
    if log_scale:
        axes[k].set_xscale("log")
    axes[k].set_title(f"{col} : Histogram")
    axes[k].grid(alpha=0.3)
    axes[k].legend()

    # quantile-bin target rate
    bins = pd.qcut(x, q=min(10, x.nunique()), duplicates="drop")
    t = (
        pd.DataFrame({"x": x, "y": y_sub, "bin": bins})
        .groupby("bin", observed=True)
        .agg(mean_y=("y", "mean"),
             count=("y", "size"),
             x_min=("x", "min"),
             x_max=("x", "max"))
        .reset_index(drop=True)
    )
    ctr = (t["x_min"] + t["x_max"]) / 2
    axes[k+1].plot(ctr, t["mean_y"], marker="o")
    axes[k+1].set_title(f"{col} : Target rate across quantile bins")
    axes[k+1].grid(alpha=0.3)
    if log_scale:
        axes[k+1].set_xscale("log")

    w = float(ctr.diff().median()) if len(ctr) > 1 else 1.0
    w = w if np.isfinite(w) and w > 0 else 1.0
    axes[k+1].twinx().bar(ctr, t["count"], alpha=0.30, width=w)

    
    
def get_kaggle_results(user, key, path, overwrite, comp, models_names):
    """Fetch Kaggle submission scores for given models and cache them to CSV."""
    if not os.path.exists(path) or overwrite:

        session = requests.Session()
        session.auth = (user, key)

        url = f"https://www.kaggle.com/api/v1/competitions/submissions/list/{comp}?page=1"
        r = session.get(url, timeout=30)

        if r.status_code == 401:
            raise SystemExit("401 Unauthorized: username/key are wrong.")
        if r.status_code == 403:
            raise SystemExit("403 Forbidden: join the competition and accept the rules on Kaggle, then retry.")
        if r.status_code == 404:
            raise SystemExit(f"404 Not Found: competition slug '{comp}' is wrong.")
        r.raise_for_status()

        perfs = pd.DataFrame(r.json())
        cols = [c for c in perfs.columns if c.lower() in {
            "ref","description","publicscore","privatescore","date","status","filename"
        }]
        perfs = perfs[cols]

        perfs["date"] = pd.to_datetime(perfs["date"], utc=True, format="mixed")
        perfs = perfs.sort_values("date", ascending=False).drop_duplicates("fileName")
        perfs["model"] = perfs["fileName"].str.extract(r"predictions_(.*)\.csv")
        perfs.index = perfs["model"]
        perfs = perfs.loc[perfs["fileName"].isin([f"predictions_{model}.csv" for model in models_names])]
        perfs = perfs.drop(["ref","date","description","status","fileName","model"], axis = 1)
        
        perfs[["publicScore","privateScore"]] = perfs[["publicScore","privateScore"]].astype(float)
        perfs.to_csv(path)

    else:
        perfs = pd.read_csv(path,index_col=0)
    return perfs

# Parse setup -> pass_variant and TE flag
def parse_setup(s):
    pv, sep, short = s.partition('_')               # split only on the first '_'
    te = short.endswith("_te")
    if te:
        short = short[:-3]
    return pv, te, short


def plot_auc_by_series(df, label_map, point_size = 110, err_capsize = 3):
    """
    Plot AUC by experimental setup, grouping lines/points by `pass_variant` and
    the boolean `te` (target encoding) flag.
    Displays the matplotlib figure (no return value).
    """

    # --- Prepare ordering, categorical x index and helper "series" column ---
    order = list(label_map.keys())

    df["series"] = df["pass_variant"].astype(str) + " | te=" + df["te"].map({True: "T", False: "F"})
    setups_order = list(df["setup"].cat.categories)

    df["setup"] = pd.Categorical(df["setup"], categories=setups_order, ordered=True)
    x_index = {s: i for i, s in enumerate(setups_order)}
    df["_x"] = df["setup"].map(x_index).astype(float)

    # --- Color, linestyle and marker mappings for variants and TE flag ---
    pass_variants = list(dict.fromkeys(df["pass_variant"].tolist()))
    palette = sns.color_palette(n_colors=len(pass_variants))
    color_map = {pv: palette[i % len(palette)] for i, pv in enumerate(pass_variants)}
    ls_map = {True: "-", False: "--"}
    marker_map = {True: "o", False: "s"}

    pv_label = {"vanilla": "No Augment", "origrow": "Rows Augments", "origcol": "Columns Augments"}
    te_label = {True: "Target Encoding", False: "No Target Encoding"}

    fig, ax = plt.subplots(figsize=(14, 12))

    # --- Plot thick faint bands, then main lines and markers for each (pass_variant, te) ---
    for pv in pass_variants:
        for te_val in [True, False]:
            sub = df[(df["pass_variant"] == pv) & (df["te"] == te_val)].sort_values("setup")

            x = sub["_x"].to_numpy()
            y = sub["auc"].to_numpy()

            # background "band" lines for aesthetic depth
            ax.plot(x, y, color=color_map[pv], linewidth=20, alpha=0.10, solid_capstyle="round", zorder=1)
            ax.plot(x, y, color=color_map[pv], linewidth=10, alpha=0.18, solid_capstyle="round", zorder=1)
            # main line (solid or dashed depending on TE)
            ax.plot(x, y, color=color_map[pv], linewidth=2.8, alpha=0.95, linestyle=ls_map[te_val], zorder=2)
            # markers for each point
            ax.scatter(
                x, y, s=point_size, marker=marker_map[te_val],
                facecolors=color_map[pv], edgecolors="white", linewidths=1.1,
                alpha=0.98, zorder=3,
            )

    # --- Axis labels and ticks ---
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([label_map[s] for s in order], rotation=25, ha="right", fontsize=13)
    ax.set_ylabel("AUC (%)", fontsize=16, labelpad=10)
    ax.set_title("AUC by Setup", pad=16, fontsize=18, weight="bold")

    # --- Y limits, major/minor locators and percentage formatting ---
    y_min, y_max = float(df["auc"].min()), float(df["auc"].max())
    pad = max((y_max - y_min) * 0.08, 0.0015)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.tick_params(axis="y", labelsize=13)

    # --- Grid and spine styling ---
    ax.grid(True, axis="y", which="major", linewidth=1.0, alpha=0.45)
    ax.grid(True, axis="y", which="minor", linewidth=0.5, alpha=0.25)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
        s.set_alpha(0.6)

    # --- Legend for pass_variant colors (bottom-left) ---
    color_handles = [
        Line2D([0],[0], color=color_map[pv], lw=6, alpha=0.9, label=pv_label.get(pv, str(pv)))
        for pv in pass_variants
    ]
    leg_colors = ax.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=len(color_handles),
        frameon=True, framealpha=0.2,
        borderpad=0.8, fontsize=14
    )
    ax.add_artist(leg_colors)

    # --- Legend for TE marker style (bottom-right) ---
    te_handles = [
        Line2D([0],[0], color="white", marker="o", linestyle="-",
               markerfacecolor="white", markeredgecolor="white",
               label=te_label[True], markersize=10),
        Line2D([0],[0], color="white", marker="s", linestyle="--",
               markerfacecolor="white", markeredgecolor="white",
               label=te_label[False], markersize=10),
    ]
    ax.legend(
        handles=te_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.15),
        ncol=len(te_handles),
        frameon=True, framealpha=0.2,
        borderpad=0.8, fontsize=14
    )

    # --- Annotate best AUC for each pass_variant (text color matches variant) ---
    idx_best = df.groupby("pass_variant")["auc"].idxmax()
    best_df = df.loc[idx_best]

    for _, r in best_df.iterrows():
        x = float(r["_x"])
        y = float(r["auc"])
        pv = r["pass_variant"]
        text = f"Best : {y*100:.2f}%"
        ax.annotate(
            text,
            (x, y),
            xytext=(0, 15), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, weight="bold",
            color=color_map[pv],   # text in pass_variant color
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc=(0, 0, 0, 0.0),   # fully transparent background
                ec="white",          # white border
                lw=1.2               # border width
            ),
            zorder=6,
        )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

    
def plot_scores(df, metrics, sort_by='roc_auc', ascending=True, top_n=None, figsize=(12,8), 
                dpi=90, title=None, vline=None, vlabel=None, legend_names=None):
    """Plot selected metrics from a DataFrame with optional sorting, labels, and reference line."""
    dfp = df.sort_values(sort_by, ascending=ascending)
    if top_n:
        dfp = dfp.tail(top_n)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = range(len(dfp))
    for m in metrics:
        if legend_names is None:
            lbl = m.replace('_',' ').title()
        elif isinstance(legend_names, dict):
            lbl = legend_names.get(m, m.replace('_',' ').title())
        else:
            lbl = legend_names[metrics.index(m)]
        ax.plot(x, dfp[m].values, marker='o', markersize=8, linestyle='--', linewidth=2,
                alpha=0.8, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(dfp.index, rotation=90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Score'); ax.legend()
    if title: ax.set_title(title)

    if vline is not None:
        ax.axvline(vline, color='white', linestyle=':', linewidth=2)
        if vlabel:
            ax.text(vline - 0.15, 0.01, vlabel, rotation=90,
                    va='bottom', ha='center', color='white',
                    transform=ax.get_xaxis_transform())
    fig.tight_layout()
    return None


def compute_all_scores(df, y, threshold = 0.5):
    """Compute a concise set of binary-classification metrics for each column in df.

    Returned metrics per model:
      - n
      - roc_auc
      - pr_auc (average precision)
      - log_loss
      - accuracy, precision, recall, f1 (computed at default threshold 0.5)
      - mcc
      - tn, fp, fn, tp
      - threshold (threshold that maximizes F1)
      - f1_opt, precision_opt, recall_opt

    Behaviour assumes well-formed numeric inputs and performs no exception handling.
    """

    y = np.asarray(y)
    n = y.shape[0]
    eps = 1e-15

    rows = []
    index = []

    for col in df.columns:
        preds = np.asarray(df[col]).astype(float)
        prob = preds
        prob_for_loss = np.clip(prob, eps, 1 - eps)

        # predictions at default threshold 0.5
        y_pred = (prob >= threshold).astype(int)

        # threshold that maximizes F1 from precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y, prob_for_loss)
        f1s = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-20)
        best_ix = int(np.argmax(f1s))
        best_threshold = float(thresholds[best_ix])
        y_pred_opt = (prob_for_loss >= best_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        metrics = {
            "n": int(n),
            "roc_auc": roc_auc_score(y, prob_for_loss),
            "pr_auc": average_precision_score(y, prob_for_loss),
            "log_loss": log_loss(y, prob_for_loss),

            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y, y_pred),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),

            "threshold": threshold,
            "f1_opt": f1_score(y, y_pred_opt, zero_division=0),
            "precision_opt": precision_score(y, y_pred_opt, zero_division=0),
            "recall_opt": recall_score(y, y_pred_opt, zero_division=0),
        }

        rows.append(metrics)
        index.append(col)

    df_score = pd.DataFrame(rows, index=index)
    df_score.index.name = "model"
    df_score = df_score.reindex(sorted(df_score.columns), axis=1)
    return df_score



# SHAP -----------

def to_numeric_codes(df: pd.DataFrame, X_num, X_cat):
    """Return a numeric-coded copy: floats for X_num, int codes for X_cat."""
    Z = df.copy()
    if X_cat:
        Z[X_cat] = Z[X_cat].apply(lambda s: s.astype("category").cat.codes.astype("int64"))
    if X_num:
        Z[X_num] = Z[X_num].astype("float64")
    return Z

def from_numeric_codes(Z, template: pd.DataFrame, X_num, X_cat):
    """
    Rebuild a DataFrame like `template` from a numeric-coded array/DataFrame Z.
    Uses template’s categorical dtypes (categories/order) and original column order.
    """
    cols = list(template.columns)
    if not isinstance(Z, pd.DataFrame):
        Z = pd.DataFrame(Z, columns=cols)

    out = {}
    # restore categoricals using template's dtype (keeps categories & order)
    for c in X_cat:
        dtype = template[c].dtype
        out[c] = pd.Categorical.from_codes(Z[c].astype("int64").to_numpy(), dtype=dtype)
    # restore numerics
    for c in X_num:
        out[c] = Z[c].astype("float64").to_numpy()

    df = pd.DataFrame(out, columns=cols)
    # enforce dtypes explicitly
    for c in X_cat:
        df[c] = df[c].astype(template[c].dtype)
    for c in X_num:
        df[c] = df[c].astype("float64")
    return df



def contains_ohe(est):
    """Recursively detect whether an estimator/transformer contains a OneHotEncoder."""
    # direct hit
    if isinstance(est, OneHotEncoder):
        return True
    
    # inside a Pipeline
    if isinstance(est, Pipeline):
        return any(contains_ohe(step) for _, step in est.steps)
    
    # inside a ColumnTransformer (before or after fit)
    if isinstance(est, ColumnTransformer):
        transformers = getattr(est, "transformers_", getattr(est, "transformers", []))
        for _, trans, _ in transformers:
            if trans in ("drop", "passthrough"):
                continue
            if contains_ohe(trans):
                return True
    return False



def compute_shap_payload(models, model_name, X, N_SAMPLE, X_num, X_cat,n_bg = 200, max_evals = 2000):
    """
    Compute SHAP Explanation for a regression Pipeline on raw features with categoricals.
    Returns a payload dict ready to cache (includes sv + sample indices + columns + meta).
    """
    # rows to explain (exact indices passed in)
    
    search = models[model_name]
    best = search.best_estimator_
    pre  = best.named_steps["pre"]
    
    has_ohe = contains_ohe(pre)

    # Decide whether we can use TreeExplainer (only when there's no OHE and the model is supported)
    can_use_tree = False
    tree_explainer = None
    if not has_ohe:
        try:
            # If this raises (e.g., InvalidModelError), we’ll fall back to the generic path
            tree_explainer = shap.TreeExplainer(best)
            can_use_tree = True
        except Exception:
            can_use_tree = False

    if has_ohe or not can_use_tree:
        X_sample_idx = X.sample(n=min(N_SAMPLE, len(X)), random_state=SEED).index
        X_sample = X.loc[X_sample_idx]


        background_raw = shap.utils.sample(X, n_bg, random_state=SEED)
        bg_num = to_numeric_codes(background_raw, X_num, X_cat)

        clust = shap.utils.hclust(bg_num.values, metric="correlation")
        masker_obj = shap.maskers.Partition(bg_num, max_samples=n_bg, clustering=clust)

        # model wrapper: accept numeric-coded, rebuild raw, call pipeline.predict
        def f_num(Z):
            raw = from_numeric_codes(Z, template=X, X_num=X_num, X_cat=X_cat)
            return best.predict(raw)  # regression → 1D

        explainer = shap.Explainer(f_num, masker_obj, algorithm="partition")

        # compute SHAP
        X_sample_num = to_numeric_codes(X_sample, X_num, X_cat)
        sv = explainer(X_sample_num, max_evals=max_evals)
    else:
        # preprocess once; we’ll pass the raw matrix to SHAP
        pre.set_output(transform="pandas")
        model = best.named_steps[model_name]
        X_pp = pre.transform(X)        
        X_sample = X_pp.sample(n=1000, random_state=SEED)
        feature_names = X_pp.columns        
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_sample, check_additivity=False)   

    return sv


# SHAP Analysis

def plot_cumulative_importance(imp, title):
    vals = imp.sort_values(ascending=False).values
    cum = np.cumsum(vals) / np.sum(vals)
    x = np.arange(1, len(vals) + 1)

    # distance from each point to line through endpoints (x1,y1)-(x2,y2)
    x1, y1 = x[0], cum[0]
    x2, y2 = x[-1], cum[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * cum + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    elbow_idx = int(np.argmax(dist))         # 0-based index into cum/x
    elbow_n = x[elbow_idx]                   # Top-N features at elbow
    elbow_val = cum[elbow_idx]               # cumulative share at elbow

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cum, linewidth=2.0)
    ax.axvline(elbow_n, linestyle='--', alpha=0.8, color = "white")
    ax.scatter([elbow_n], [elbow_val], color = "red")
    ax.annotate(f"elbow: {elbow_n} ({elbow_val*100:.1f}%)",
                xy=(elbow_n, elbow_val), xytext=(10, -20), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", alpha=0.6))
    ax.set_xlabel("Top-N features")
    ax.set_ylabel("Cumulative share of total Mean |SHAP|")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    
def compute_percentage(df, prefixes, max_display = 15, collapse_twins = True):

    # ---- Aggregate imp_df to base features with 3×3 kinds ----
    base_totals = {}
    base_breakdown = {}

    for feat, imp in df.items():
        prefix, core = "", feat
        for p in prefixes:
            if feat.startswith(p):
                prefix, core = p.rstrip("_"), feat[len(p):]

        if "_" in core:
            a, b = core.split("_", 1)
            parts = (a, b)
            weight_each = 0.5
            kind_type = "Pair"
        else:
            parts = (core,)
            weight_each = 1.0
            kind_type = "Twin" if core.endswith("2") else "Raw"

        family = "Base"
        if prefix in ("TE", "TE_ORIG"):
            family = "TE"
        if prefix == "CE":
            family = "CE"
        kind = f"{family}-{kind_type}"

        for p in parts:
            base = p
            if collapse_twins and base.endswith("2"):
                base = base[:-1] 

            base_totals[base] = base_totals.get(base, 0.0) + imp * weight_each
            if base not in base_breakdown:
                base_breakdown[base] = {}
            base_breakdown[base][kind] = base_breakdown[base].get(kind, 0.0) + imp * weight_each

    base_imp = pd.Series(base_totals).sort_values(ascending=True)
    base_imp = base_imp[-max_display:] if len(base_imp) > max_display else base_imp
    labels = list(base_imp.index)

    # Build stacked matrix in % of global total
    total_global = float(base_imp.sum())
    stack_rows = []
    for base in labels:
        row = [base_breakdown.get(base, {}).get(k, 0.0) for k in colors.keys()]
        stack_rows.append(row)
    stack = np.array(stack_rows)
    stack_pct = stack / total_global
    totals_pct = base_imp.values / total_global 
    
    return stack_pct, totals_pct, labels

# Color palette (ordered - determines stack order)
colors = {
    "Base-Raw":  "#1b9e77",  # green (dark)
    "Base-Twin": "#66c2a5",  # green (medium)
    "Base-Pair": "#a6dba0",  # green (light)
    "CE-Raw":    "#ffd92f",  # yellow
    "CE-Twin":   "#fc8d59",  # orange
    "CE-Pair":   "#d73027",  # red
    "TE-Raw":    "#c6dbef",  # blue (light)
    "TE-Twin":   "#6baed6",  # blue (medium)
    "TE-Pair":   "#1f78b4",  # blue (dark)
}

def plot_per_splits(totals, stack, labels):
    plt.figure(figsize=(12.5, max(4, 0.55 * len(labels))))
    ax = plt.gca()
    left = np.zeros(len(labels))
    for j, kind in enumerate(colors.keys()):
        ax.barh(labels, stack[:, j], left=left, label=kind, color=colors[kind])
        left += stack[:, j]

    xmax = float(max(totals))
    padding = max(0.03, 0.12 * xmax)
    xlim_right = min(1.0 + padding, xmax + padding)
    ax.set_xlim(0, xlim_right)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    ax.set_xlabel("Share of global Mean |SHAP|", fontsize=14)
    ax.set_title("Base Features - Composition by Source (as % of global importance)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc="lower right", ncol=3, frameon=False, prop={'size': 14})

    for i, pct in enumerate(totals):
        x_text = pct + padding * 0.1
        if x_text > xlim_right * 0.995:
            x_text = xlim_right * 0.995
        ax.text(x_text, i, f"{pct:.2%}", va="center", ha="left", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.06)
    plt.show()
    

def plot_aggregate(df, prefixes):
    kind_sums = dict.fromkeys(colors.keys(), 0.0)
    for feat, imp in df.items():
        for p in prefixes:
            if feat.startswith(p):
                prefix = p.rstrip("_")
                core = feat[len(p):]
                break
        else:
            prefix, core = "", feat
        fam = "TE" if prefix in ("TE", "TE_ORIG") else ("CE" if prefix == "CE" else "Base")
        typ = "Pair" if "_" in core else ("Twin" if core.endswith("2") else "Raw")
        kind_sums[f"{fam}-{typ}"] += imp

    kinds = list(colors.keys())
    vals = np.array([kind_sums[k] for k in kinds])
    order = np.argsort(-vals)
    kinds = [kinds[i] for i in order]
    vals = vals[order]
    cols = [colors[k] for k in kinds]

    vals_pct = vals / vals.sum() 
    pad = max(0.03, 0.12 * vals_pct.max())
    ylim_top = min(1.0 + pad, (vals_pct.max()) + pad)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    x = np.arange(len(kinds))
    ax.bar(x, vals_pct, color=cols)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=45, ha='right', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of global Mean |SHAP|", fontsize=14)
    ax.set_ylim(0, ylim_top)

    for i, v in enumerate(vals_pct):
        ax.text(i, v + pad * 0.03, f"{v:.2%}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.show()