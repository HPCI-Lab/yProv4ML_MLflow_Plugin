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
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
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

    # 1) Drop fully empty rows
    df = df.dropna(how="all")

    # 2) Remove accidental repeated header rows (row equals column names)
    try:
        header = list(df.columns.astype(str))
        mask_is_header_row = df.astype(str).apply(lambda r: list(r.values) == header, axis=1)
        if mask_is_header_row.any():
            df = df.loc[~mask_is_header_row]
    except Exception:
        pass

    # 3) Strip whitespace from string cells (helps de-dup)
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)

    # 4) Drop exact duplicate rows
    df = df.drop_duplicates()

    # 5) Prefer uniqueness by run identifier if available
    if "run_id" in df.columns:
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
acc_cands = candidate_accuracy_cols(df)
cost_cands = [c for c in candidate_cost_cols(df) if c != id_col]  # don't suggest ID as cost

st.sidebar.header("2) Metrics & settings")
acc_col = st.sidebar.selectbox("Accuracy metric (maximize)", options=(acc_cands or ncols), index=0)
cost_col = st.sidebar.selectbox("Cost/Emission metric (minimize)", options=(cost_cands or ncols), index=0)

st.sidebar.subheader("Clustering")
clustering_mode = st.sidebar.radio("Cluster on:", ["Full dataset", "Pareto-only"])
n_clusters = st.sidebar.slider("KMeans: number of clusters", 2, 8, 3)
annotate = st.sidebar.checkbox("Label points on plots", value=True)

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
labels = None

if SKLEARN_OK and len(cluster_df) >= n_clusters:
    model, labels = kmeans_cluster(cluster_df, [acc_col, cost_col], n_clusters=n_clusters)
    if labels is not None:
        cluster_df = cluster_df.copy()
        cluster_df["cluster"] = labels
        st.write(f"**KMeans clusters** on {clustering_mode} set: K={n_clusters}")

        cluster_df["cluster_label"] = cluster_df["cluster"].astype(str)
        PALETTE = px.colors.qualitative.Set2

        hover = cluster_df.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
        figc = px.scatter(
            cluster_df,
            x=cost_col, y=acc_col,
            color="cluster_label",
            color_discrete_sequence=PALETTE,
            hover_name=id_col if (id_col and id_col in cluster_df.columns) else None,
            hover_data=[c for c in pcols if c in cluster_df.columns],
            title=f"Clusters on {clustering_mode.lower()} set"
        )
        figc.update_layout(legend_title_text="cluster", xaxis_title=cost_col, yaxis_title=acc_col)
        if annotate and id_col and (id_col in cluster_df.columns):
            figc.update_traces(text=cluster_df[id_col].astype(str), textposition="top center", selector=dict(mode="markers"))
        st.plotly_chart(figc, use_container_width=True)

        with st.expander("Cluster-wise summary (mean/min/max/count)"):
            summary = summarize_clusters(cluster_df, "cluster", [acc_col, cost_col])
            st.dataframe(summary)
    else:
        st.info("Clustering skipped (insufficient rows).")
else:
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed; clustering disabled. `pip install scikit-learn`")
    else:
        st.info("Not enough rows for clustering with current selection.")

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

