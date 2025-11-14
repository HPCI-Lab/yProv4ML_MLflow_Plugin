# decision_app.py
# Streamlit Decision-Support Dashboard for yProv4ML experiments (single CSV upload)
#
# Run:
#   streamlit run decision_app.py
#
# Features
# - Upload one CSV (no dataset names, no defaults)
# - Auto-detect id column, param_* columns, numeric metrics
# - Select accuracy (maximize) and cost (minimize)
# - Pareto front scatter (with labels), KMeans clustering (optional)
# - Descriptive stats, correlations
# - Prescriptive suggestions (parameter ranges + next experiments)
#
# Requires: streamlit, pandas, numpy, plotly, (optional) scikit-learn

import io
import re
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------- Optional deps -----------------------------


try:
    import shap
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.preprocessing import StandardScaler  # used for distance-based method
    SHAP_OK = True
    SKLEARN_OK = True
except Exception:
    SHAP_OK = False
    SKLEARN_OK = False




# ----------------------------- Cache loaders -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_bytes(buf: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(buf))

# ----------------------------- Helpers -----------------------------
def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["exp", "run_id", "id", "experiment", "run"]:
        if c in df.columns:
            return c
    return None

def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def param_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("param_")]

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Keep first occurrence if a column name repeats
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all")
    try:
        header = list(df.columns.astype(str))
        mask_is_header_row = df.astype(str).apply(
            lambda r: list(r.values) == header, axis=1
        )
        if mask_is_header_row.any():
            df = df.loc[~mask_is_header_row]
    except Exception:
        pass

    # 3) Strip whitespace from string cells
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)

    # 4) Drop exact duplicate rows
    df = df.drop_duplicates()

    # 5) Prefer uniqueness by run identifier *only if it makes sense*
    if "run_id" in df.columns:
        col = df["run_id"]
        # non-null count and unique non-null count
        non_null = col.notna().sum()
        nunique = col.nunique(dropna=True)
        # Only dedup if there ARE non-null values AND actually repeated IDs
        if non_null > 0 and nunique < non_null:
            df = df.drop_duplicates(subset=["run_id"], keep="last")

    return df.reset_index(drop=True)



def candidate_accuracy_cols(df: pd.DataFrame) -> List[str]:
    cands = [c for c in df.columns if re.search(r"(acc|accuracy)", c, re.I)]
    if "ACC_val" in cands:
        cands.remove("ACC_val")
        cands = ["ACC_val"] + cands
    # fallback: first few numeric columns not named *loss*
    return cands or [c for c in numeric_cols(df) if "loss" not in c.lower()][:5]

def candidate_cost_cols(df: pd.DataFrame) -> List[str]:
    keys = ["emission", "co2", "energy", "time", "latency", "cost", "power"]
    cands = [c for c in df.columns if any(k in c.lower() for k in keys)]
    order = ["emissions", "emissions_gCO2eq", "energy_consumed", "energy_J", "train_epoch_time_ms"]
    ranked = [c for c in order if c in df.columns]
    ranked += [c for c in cands if c not in ranked]
    return ranked or numeric_cols(df)

def _safe_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return 1-D Series even if duplicate column names exist."""
    col = df[col_name]
    return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

def nondominated_front(df: pd.DataFrame, acc_col: str, cost_col: str) -> pd.DataFrame:
    """Maximize accuracy, minimize cost."""
    arr_acc = _safe_series(df, acc_col).to_numpy()
    arr_cost = _safe_series(df, cost_col).to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dom = (arr_acc >= arr_acc[i]) & (arr_cost <= arr_cost[i]) & (
            (arr_acc > arr_acc[i]) | (arr_cost < arr_cost[i])
        )
        if np.any(dom & (np.arange(len(df)) != i)):
            keep[i] = False
    return df.loc[keep]

def summarize_clusters(df: pd.DataFrame, cluster_col: str, metrics: List[str]) -> pd.DataFrame:
    if cluster_col not in df.columns:
        return pd.DataFrame()
    agg = {}
    for m in metrics:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            agg[m] = ["count", "mean", "min", "max"]
    if not agg:
        return pd.DataFrame()
    return df.groupby(cluster_col).agg(agg)

def kmeans_cluster(df: pd.DataFrame, cols: List[str], n_clusters: int, random_state: int = 42):
    if not SKLEARN_OK or len(df) < max(2, n_clusters):
        return None, None
    X = df[cols].to_numpy()
    try:
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    except TypeError:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels

def suggest_parameter_ranges(df: pd.DataFrame, good_mask: np.ndarray, param_columns: List[str]) -> Dict[str, Tuple[float, float]]:
    recs = {}
    good_df = df.loc[good_mask]
    for p in param_columns:
        if p in good_df.columns:
            series = good_df[p].dropna()
            if len(series) == 0:
                continue
            try:
                lo, hi = float(series.min()), float(series.max())
                recs[p] = (lo, hi)
            except Exception:
                top = series.value_counts().index.tolist()[:3]
                recs[p] = ("categorical", top)
    return recs

def _scalar_from_row(row: pd.Series, key: Optional[str]):
    if not key or key not in row.index:
        return None
    v = row[key]
    return v.iloc[0] if isinstance(v, pd.Series) else v

def make_tooltip_text(row: pd.Series, id_col: Optional[str], param_columns: List[str], acc_col: str, cost_col: str) -> str:
    rid_val = _scalar_from_row(row, id_col)
    rid = str(rid_val) if rid_val is not None else f"(idx {row.name})"
    params = ", ".join([f"{p}={_scalar_from_row(row, p)}" for p in param_columns if p in row.index])
    a = _scalar_from_row(row, acc_col); c = _scalar_from_row(row, cost_col)
    a_txt = f"{a:.4g}" if (a is not None and pd.notnull(a) and np.isfinite(float(a))) else str(a)
    c_txt = f"{c:.4g}" if (c is not None and pd.notnull(c) and np.isfinite(float(c))) else str(c)
    return f"id={rid}<br>{acc_col}={a_txt}<br>{cost_col}={c_txt}<br>{params}"

def neighbor_grid(val, scale=0.5, n=3):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return []
    try:
        v = float(val)
        return sorted(set([v, v*(1-scale), v*(1+scale)]))[:n]
    except Exception:
        return [val]
    
def run_clustering(df_in: pd.DataFrame, x_col: str, y_col: str, method: str, params: dict):
    """
    Cluster on two selected metrics (x=cost, y=accuracy) with different algorithms.
    Returns (labels or None, fitted_model or None, df_used)
    """
    # Only rows that have both metrics
    df_used = df_in.dropna(subset=[x_col, y_col]).copy()
    if df_used.empty:
        return None, None, df_used

    X = df_used[[x_col, y_col]].to_numpy()

    # Scale for distance-based clustering (DBSCAN/Agglo/Spectral). KMeans is fine either way.
    scale_needed = method in {"DBSCAN", "Agglomerative", "Spectral"}
    if scale_needed:
        try:
            X = StandardScaler().fit_transform(X)
        except Exception:
            # simple fallback standardization
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    model = None
    labels = None

    if method == "KMeans":
        k = int(params.get("k", 3))
        init = params.get("init", "k-means++")
        try:
            model = KMeans(n_clusters=k, n_init="auto", init=init, random_state=42)
        except TypeError:
            model = KMeans(n_clusters=k, n_init=10, init=init, random_state=42)
        labels = model.fit_predict(X)

    elif method == "DBSCAN":
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        # DBSCAN labels: -1 = noise; map noise to its own cluster label string later in plots

    elif method == "Agglomerative":
        k = int(params.get("k", 3))
        linkage = params.get("linkage", "ward")  # 'ward'| 'complete' | 'average' | 'single'
        # Ward requires euclidean and n_clusters > 1
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)

    elif method == "Spectral":
        k = int(params.get("k", 3))
        assign = params.get("assign_labels", "kmeans")  # 'kmeans' or 'discretize'
        model = SpectralClustering(n_clusters=k, assign_labels=assign, random_state=42, affinity="rbf")
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, model, df_used

    
def get_param_feature_types(df: pd.DataFrame, pcols: list):
    """Split param columns into numeric vs categorical."""
    pcols = [c for c in pcols if c in df.columns]
    num_cols, cat_cols = [], []
    for c in pcols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def build_surrogate_pipeline(model_key: str, num_cols: list, cat_cols: list):
    """Return a sklearn Pipeline with preprocessing + chosen classifier."""
    # Choose model
    if model_key == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    elif model_key == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unknown surrogate model: {model_key}")

    # Preprocess
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    if not transformers:
        # fallback: no params? will be handled upstream
        pre = "passthrough"
    else:
        pre = ColumnTransformer(transformers, remainder="drop")

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def get_feature_names_after_pre(proc: ColumnTransformer, num_cols: list, cat_cols: list):
    """Get names after ColumnTransformer for SHAP labeling."""
    names = []
    if num_cols:
        names.extend(num_cols)
    if cat_cols:
        # pull OneHotEncoder categories
        ohe = None
        for name, trans, cols in proc.transformers_:
            if name == "cat":
                ohe = trans
                used_cols = cols
                break
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            names.extend(list(ohe.get_feature_names_out(used_cols)))
        else:
            # fallback generic names
            for c in cat_cols:
                names.append(f"{c}_encoded")
    return names or None

def compute_shap_values(pipe: Pipeline, X: pd.DataFrame):
    """Return (explainer, shap_values, is_multiclass)."""
    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    Xp = pre.transform(X)

    # Pick explainer by model family
    try:
        if isinstance(clf, RandomForestClassifier):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(Xp)
        elif isinstance(clf, LogisticRegression):
            # Use a small background for stability
            bg = shap.sample(Xp, min(200, Xp.shape[0])) if Xp.shape[0] > 200 else Xp
            explainer = shap.LinearExplainer(clf, bg)
            shap_values = explainer.shap_values(Xp)
        else:
            # Generic fallback
            bg = shap.sample(Xp, min(200, Xp.shape[0])) if Xp.shape[0] > 200 else Xp
            explainer = shap.KernelExplainer(clf.predict_proba, bg)
            shap_values = explainer.shap_values(Xp)
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}")

    multiclass = isinstance(shap_values, list)
    return explainer, shap_values, multiclass, Xp


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="yProv4ML Decision Support", layout="wide")
st.title("yProv4ML • Decision-Support Dashboard")
st.write("Upload a CSV to analyze **performance vs sustainability** trade-offs (Pareto, clustering, prescriptions).")

# ----------------------------- Upload CSV (single source) -----------------------------
st.sidebar.header("1) Upload CSV")
up = st.sidebar.file_uploader("Select a CSV file", type=["csv"], key="single_csv")

df = None
if up is not None:
    try:
        raw = load_csv_from_bytes(up.getvalue())
        raw = ensure_unique_columns(raw)
        df = clean_rows(raw)
        st.sidebar.success(f"Loaded file: {up.name} (rows={len(df)})")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

with st.expander("Debug: first rows & shape"):
    st.write(df.shape)
    st.dataframe(df.head(10))


if df is None:
    st.info("Please upload a CSV in the left sidebar to continue.")
    st.stop()

# ----------------------------- Structure detection -----------------------------
id_col = detect_id_column(df)
pcols = param_cols(df)
ncols = numeric_cols(df)

# helper: filter out anything ID-like
ID_EXACT = {"id", "run_id", "run", "exp", "experiment"}
def is_id_like(name: str) -> bool:
    n = str(name).strip().lower()
    return (n in ID_EXACT) or n.endswith("_id")

def drop_id_like(cols):
    return [c for c in cols if not is_id_like(c)]

acc_cands_raw  = candidate_accuracy_cols(df)
cost_cands_raw = [c for c in candidate_cost_cols(df) if c != id_col]  # existing guard

# Apply ID-like filter everywhere
acc_cands  = drop_id_like(acc_cands_raw)
cost_cands = drop_id_like(cost_cands_raw)
ncols_no_id = drop_id_like(ncols)

# Neutral = numeric metrics that aren't in either list and aren't params or ID-like
neutral = [
    c for c in ncols_no_id
    if c not in set(acc_cands) | set(cost_cands)
    and not str(c).startswith("param_")
]

# Show neutral candidates in BOTH dropdowns (dedup while preserving order)
acc_options  = list(dict.fromkeys(acc_cands  + neutral))
cost_options = list(dict.fromkeys(cost_cands + neutral))

st.sidebar.header("2) Metrics & settings")
acc_col  = st.sidebar.selectbox(
    "Accuracy metric (maximize)",
    options=(acc_options or ncols_no_id), index=0
)
cost_col = st.sidebar.selectbox(
    "Cost/Emission metric (minimize)",
    options=(cost_options or ncols_no_id), index=0
)



st.sidebar.subheader("Clustering")
clustering_mode = st.sidebar.radio("Cluster on:", ["Full dataset", "Pareto-only"])
n_clusters = st.sidebar.slider("KMeans: number of clusters", 2, 8, 3)
annotate = st.sidebar.checkbox("Label points on plots", value=True)
st.sidebar.subheader("Cluster explanations (SHAP)")
surrogate_model = st.sidebar.selectbox(
    "Surrogate model",
    options=["Random Forest", "Logistic Regression"],
    index=0,
    key="surrogate_model"
)


# ----------------------------- Filter rows for selected metrics -----------------------------
work = df.dropna(subset=[acc_col, cost_col]).copy()
if len(work) < 2:
    st.warning("Not enough rows with both selected metrics.")
    st.stop()

# ----------------------------- Descriptive analytics -----------------------------
st.subheader("Descriptive analytics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric(f"{acc_col} (mean)", f"{df[acc_col].mean():.4g}")
with col3:
    st.metric(f"{cost_col} (mean)", f"{df[cost_col].mean():.4g}")


with st.expander("Correlation (Spearman) among numeric columns"):
    if len(ncols) >= 2:
        corr = work[ncols].corr(method="spearman")
        st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None))
    else:
        st.info("Not enough numeric columns to compute a correlation matrix.")

# ----------------------------- Pareto analysis -----------------------------
st.subheader("Pareto analysis")
cols_for_front = [acc_col, cost_col] + pcols
if id_col and id_col not in cols_for_front:
    cols_for_front.append(id_col)
cols_for_front = [c for c in cols_for_front if c in work.columns]  # guard + dedupe

front = nondominated_front(work[cols_for_front], acc_col, cost_col)
st.write(f"Found **{len(front)}** Pareto-optimal runs (maximize `{acc_col}`, minimize `{cost_col}`).")

def plot_scatter_all_vs_front(df_all, df_front, acc_col, cost_col, id_col, pcols, title, annotate_points: bool):
    fig = go.Figure()
    hover_all = df_all.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
    fig.add_trace(go.Scatter(
        x=df_all[cost_col], y=df_all[acc_col],
        mode="markers", name="All runs",
        opacity=0.35, marker=dict(size=9, line=dict(width=0.5, color="black")),
        text=hover_all, hoverinfo="text"
    ))
    hover_front = df_front.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
    fig.add_trace(go.Scatter(
        x=df_front[cost_col], y=df_front[acc_col],
        mode="markers+text" if annotate_points else "markers",
        name="Pareto front", marker=dict(size=12, line=dict(width=1, color="black")),
        text=[str(_scalar_from_row(r, id_col)) if (annotate_points and id_col and (id_col in r.index)) else None
              for _, r in df_front.iterrows()],
        textposition="top center",
        hovertext=hover_front, hoverinfo="text"
    ))
    fig.update_layout(
        title=title,
        xaxis_title=cost_col,
        yaxis_title=acc_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

st.plotly_chart(
    plot_scatter_all_vs_front(work, front, acc_col, cost_col, id_col, pcols,
                              "All runs vs Pareto front", annotate),
    use_container_width=True
)

# ----------------------------- Clustering -----------------------------
st.subheader("Clustering")
cluster_df = front if clustering_mode == "Pareto-only" else work

# Sidebar controls for method & params
st.sidebar.subheader("Clustering")
method = st.sidebar.selectbox("Method", ["KMeans", "DBSCAN", "Agglomerative", "Spectral"], index=0)

# Shared + per-method controls
if method == "KMeans":
    k = st.sidebar.slider("K (clusters)", 2, 12, 3)
    init = st.sidebar.selectbox("Init", ["k-means++", "random"], index=0)
    params = {"k": k, "init": init}
elif method == "DBSCAN":
    eps = st.sidebar.slider("eps", 0.05, 5.0, 0.5, 0.05)
    min_samples = st.sidebar.slider("min_samples", 2, 50, 5, 1)
    params = {"eps": eps, "min_samples": min_samples}
elif method == "Agglomerative":
    k = st.sidebar.slider("K (clusters)", 2, 12, 3)
    linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)
    params = {"k": k, "linkage": linkage}
else:  # Spectral
    k = st.sidebar.slider("K (clusters)", 2, 12, 3)
    assign_labels = st.sidebar.selectbox("Assign labels", ["kmeans", "discretize"], index=0)
    params = {"k": k, "assign_labels": assign_labels}

annotate = st.sidebar.checkbox("Label points on plots", value=True, key="annotate_clusters")

labels, model, used = (None, None, cluster_df)
if SKLEARN_OK and len(cluster_df) >= 2:
    try:
        labels, model, used = run_clustering(cluster_df, cost_col, acc_col, method, params)
    except Exception as e:
        st.warning(f"Clustering failed: {e}")

if labels is None:
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed; clustering disabled. `pip install scikit-learn`")
    else:
        st.info("Not enough rows or parameters for clustering with current selection.")
else:
    used = used.copy()
    used["cluster"] = labels
    # Normalize label names for legend (DBSCAN uses -1 for noise)
    used["cluster_label"] = used["cluster"].apply(lambda z: "noise" if z == -1 else str(z))

    st.write(f"**{method}** on {clustering_mode} set")

    PALETTE = px.colors.qualitative.Set2
    hover = used.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
    figc = px.scatter(
        used,
        x=cost_col, y=acc_col,
        color="cluster_label",
        color_discrete_sequence=PALETTE,
        hover_name=id_col if (id_col and id_col in used.columns) else None,
        hover_data=[c for c in pcols if c in used.columns],
        title=f"{method} clusters on {clustering_mode.lower()} set"
    )
    figc.update_layout(legend_title_text="cluster", xaxis_title=cost_col, yaxis_title=acc_col)
    if annotate and id_col and (id_col in used.columns):
        figc.update_traces(text=used[id_col].astype(str), textposition="top center", selector=dict(mode="markers"))
    st.plotly_chart(figc, use_container_width=True)

    with st.expander("Cluster-wise summary (mean/min/max/count)"):
        summary = summarize_clusters(used, "cluster", [acc_col, cost_col])
        st.dataframe(summary)

    # Keep a consistent `cluster_df` for SHAP section downstream
    cluster_df = used

# ----------------------------- Cluster explanations (SHAP) -----------------------------
st.subheader("Cluster explanations (SHAP)")

# 1) Features to use (params only)
p_num, p_cat = get_param_feature_types(cluster_df, pcols)
used_pcols = p_num + p_cat
if not used_pcols:
    st.info("No `param_*` columns available to explain clusters.")
else:
    # Drop constant params (no signal)
    varying_pcols = [c for c in used_pcols if cluster_df[c].nunique(dropna=True) > 1]
    if not varying_pcols:
        st.info("All parameter columns are constant across selected runs; no signal to explain clusters.")
    else:
        X = cluster_df[varying_pcols].copy()
        y = cluster_df["cluster"].astype(int).values

        # Need at least 2 clusters
        class_counts = pd.Series(y).value_counts().sort_index()
        if class_counts.shape[0] < 2:
            st.info("Clustering produced a single label. Need at least 2 clusters to compute explanations.")
        else:
            # 2) Fit surrogate
            try:
                pipe = build_surrogate_pipeline(
                    surrogate_model,
                    [c for c in varying_pcols if c in p_num],
                    [c for c in varying_pcols if c in p_cat]
                )
                pipe.fit(X, y)
            except Exception as e:
                pipe = None
                st.error(f"Could not fit surrogate model: {e}")

            if pipe is not None:
                # 3) Feature names after preprocessing
                try:
                    pre = pipe.named_steps["pre"]
                    f_names = get_feature_names_after_pre(
                        pre,
                        [c for c in varying_pcols if c in p_num],
                        [c for c in varying_pcols if c in p_cat]
                    )
                except Exception:
                    f_names = varying_pcols

                # 4) Try SHAP first
                importances = None
                try:
                    explainer, shap_values, multiclass, Xp = compute_shap_values(pipe, X)
                    Xp = np.asarray(Xp)
                    if np.any(~np.isfinite(Xp)):
                        Xp = np.nan_to_num(Xp, nan=0.0, posinf=1e9, neginf=-1e9)

                    if multiclass:
                        importances = np.mean(np.stack([np.abs(sv) for sv in shap_values], axis=0), axis=(0, 1))
                    else:
                        importances = np.mean(np.abs(shap_values), axis=0)
                except Exception as e:
                    st.caption(f"⚠️ SHAP could not be computed: {e}")
                    importances = None

                # 5) Fallback to model-native importances if SHAP missing/zero
                fallback_used = False
                if (importances is None) or (float(np.max(np.abs(importances))) == 0.0):
                    clf = pipe.named_steps["clf"]
                    if hasattr(clf, "feature_importances_"):
                        importances = np.asarray(clf.feature_importances_)
                        fallback_used = True
                    elif hasattr(clf, "coef_"):
                        coef = np.asarray(clf.coef_)
                        importances = (np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef).ravel())
                        fallback_used = True
                    else:
                        importances = None

                # 6) Plot global importance (or show message)
                if importances is None:
                    st.info("Surrogate model provides no importances; cannot explain clusters.")
                else:
                    # Normalize names length to #features
                    n_feats = int(len(importances))
                    if not isinstance(f_names, list) or len(f_names) != n_feats:
                        f_names_list = [f"f{i}" for i in range(n_feats)]
                    else:
                        f_names_list = list(f_names)

                    order = np.argsort(importances)[::-1].ravel().tolist()
                    names_sorted = [f_names_list[int(i)] for i in order]
                    imps_sorted = np.asarray(importances)[order].ravel()

                    # Dynamic slider bounded to available features
                    k = st.slider(
                        "Top features to show",
                        min_value=1,
                        max_value=int(n_feats),
                        value=min(12, int(n_feats)),
                        key="shap_topk_dynamic"
                    )
                    names_k = names_sorted[:k]
                    imps_k = imps_sorted[:k].tolist()

                    title_note = " (model-native importances)" if fallback_used else " (mean |SHAP|)"
                    st.markdown(f"**Global feature importance** · Surrogate: _{surrogate_model}_{title_note}")
                    try:
                        fig_bar = px.bar(
                            x=imps_k[::-1],
                            y=names_k[::-1],
                            orientation="h",
                            title=f"Top {k} features",
                            labels={"x": "importance", "y": "feature"}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    except Exception:
                        st.write(pd.DataFrame({"feature": names_k, "importance": imps_k}))



# ----------------------------- Prescriptive analytics -----------------------------
st.subheader("Prescriptive analytics")

q_acc = st.slider("Top-quantile for accuracy (good ≥)", 0.50, 0.95, 0.80, 0.05)
q_cost = st.slider("Bottom-quantile for cost (good ≤)", 0.05, 0.50, 0.25, 0.05)
thr_acc = work[acc_col].quantile(q_acc)
thr_cost = work[cost_col].quantile(q_cost)
st.write(f"Thresholds → **{acc_col} ≥ {thr_acc:.4g}** and **{cost_col} ≤ {thr_cost:.4g}**.")

good_mask = (work[acc_col] >= thr_acc) & (work[cost_col] <= thr_cost)
st.write(f"Selected **{int(good_mask.sum())}/{len(work)}** runs as *high-utility* candidates.")

if good_mask.any():
    cols_view = [acc_col, cost_col] + ([id_col] if id_col else []) + pcols
    cols_view = [c for c in cols_view if c in work.columns]
    good = work.loc[good_mask, cols_view].copy()
    st.dataframe(good.sort_values([cost_col, acc_col], ascending=[True, False]).head(20))

    recs = suggest_parameter_ranges(work, good_mask.values, pcols)
    if recs:
        st.markdown("**Recommended parameter ranges / choices** (from high-utility runs):")
        for p, v in recs.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "categorical":
                st.write(f"- `{p}`: prefer values **{v[1]}**")
            elif isinstance(v, tuple) and len(v) == 2:
                st.write(f"- `{p}`: try range **[{v[0]:.4g}, {v[1]:.4g}]**")
    st.markdown("**Next experiments (around frontier)**")
    sort_cols = [c for c in [cost_col, acc_col] if c in front.columns]
    top_front = front.sort_values(sort_cols, ascending=[True, False]).head(5)
    for idx, row in top_front.iterrows():
        rid = row[id_col] if id_col and (id_col in row) else idx
        a = row.get(acc_col, None); c = row.get(cost_col, None)
        a_txt = f"{a:.4g}" if pd.notnull(a) and np.isfinite(a) else str(a)
        c_txt = f"{c:.4g}" if pd.notnull(c) and np.isfinite(c) else str(c)
        st.write(f"- Front run **{rid}**: {acc_col}={a_txt}, {cost_col}={c_txt}")
        proposals = {}
        for p in pcols:
            val = row.get(p, None)
            if val is None:
                continue
            if isinstance(val, (int, np.integer)):
                proposals[p] = [int(val)]
            else:
                proposals[p] = neighbor_grid(val, scale=0.5)
        st.code(json.dumps(proposals, indent=2))
else:
    st.info("No runs meet both thresholds. Try adjusting the sliders or switch cost metric.")

st.caption("Built for human-in-the-loop, multi-objective model selection with yProv4ML.")

