# decision_app_enhanced_v2.py
# Enhanced Streamlit Decision-Support Dashboard for yProv4ML experiments
#
# NEW in v2:
# - Fixed overlapping axis labels in pairwise analysis
# - SHAP importance analysis for clustering
# - Surrogate model selection for SHAP
# - Better layout and spacing
#
# Run:
#   streamlit run decision_app_enhanced_v2.py
#
# Requires: streamlit, pandas, numpy, plotly, scikit-learn, shap

import io
import re
import json
from typing import List, Tuple, Optional, Dict
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------- Optional deps -----------------------------

try:
    import shap
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
    """Heuristic detection of hyperparameter columns."""
    # 1) Explicit param_* columns win
    explicit = [c for c in df.columns if str(c).lower().startswith("param_")]
    if explicit:
        return explicit

    n_rows = len(df)
    if n_rows == 0:
        return []

    # Negative filters
    id_exact = {"id", "run_id", "run", "exp", "experiment"}
    id_suffixes = ["_id"]
    metric_keywords = ["acc", "accuracy", "loss", "emission", "co2", "energy",
                      "time", "latency", "cost", "power", "score", "metric"]
    path_keywords = ["path", "file", "dir", "uri", "artifact", "json", "requirements"]
    dataset_keywords = ["dataset", "usecase"]

    # Positive hints for hyperparameters
    param_hints = ["lr", "learning_rate", "batch", "bs", "epoch", "dropout",
                   "width", "depth", "hidden", "layer", "layers", "heads", "nhead",
                   "weight_decay", "wd", "momentum", "alpha", "beta", "gamma", "lambda",
                   "model_size", "size"]

    candidates: List[str] = []

    for c in df.columns:
        name = str(c).strip().lower()

        # Skip IDs
        if name in id_exact or any(name.endswith(suf) for suf in id_suffixes):
            continue

        # Skip metrics
        if any(k in name for k in metric_keywords):
            continue

        # Skip paths
        if any(k in name for k in path_keywords):
            continue

        # Skip dataset/usecase
        if any(k in name for k in dataset_keywords):
            continue

        col = df[c]
        nunique = col.nunique(dropna=True)

        # Skip constants
        if nunique <= 1:
            continue

        # Check if param-like by name
        is_param_by_name = any(h in name for h in param_hints)

        # Cardinality heuristic
        max_allowed = max(20, int(0.5 * n_rows)) if n_rows > 1 else nunique
        is_param_by_cardinality = (nunique <= max_allowed)

        if is_param_by_name or is_param_by_cardinality:
            candidates.append(c)

    return candidates

def create_correlation_heatmap(df: pd.DataFrame, pcols: List[str], acc_col: str, cost_col: str):
    """
    Create a correlation heatmap between numeric hyperparameters
    and the selected metrics (accuracy & cost).
    """
    # Numeric hyperparameters with variation
    numeric_params = [
        p for p in pcols
        if p in df.columns
        and pd.api.types.is_numeric_dtype(df[p])
        and df[p].nunique() >= 2
    ]

    # Add metrics (if numeric)
    cols = numeric_params.copy()
    for m in [acc_col, cost_col]:
        if m in df.columns and m not in cols and pd.api.types.is_numeric_dtype(df[m]):
            cols.append(m)

    # Need at least 2 columns for correlation
    if len(cols) < 2:
        return None

    corr = df[cols].corr(method="pearson")

    # Labels as strings
    x_labels = list(corr.columns)
    y_labels = list(corr.index)

    # Round for display text
    text_vals = np.round(corr.values, 2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "black"},
            hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="Pearson r",
                len=0.8,
            ),
            xgap=1,
            ygap=1,
        )
    )

    fig.update_xaxes(
        tickangle=-45,
        title_text="Features",
        title_font=dict(size=12),
        tickfont=dict(size=11),
        type="category",
        categoryorder="array",
        categoryarray=x_labels,
    )

    fig.update_yaxes(
        title_text="Features",
        title_font=dict(size=12),
        tickfont=dict(size=11),
        type="category",
        categoryorder="array",
        categoryarray=y_labels,
    )

    fig.update_layout(
        title_text="Correlation between Hyperparameters and Metrics",
        height=500,
        margin=dict(t=80, b=80, l=100, r=80),
        showlegend=False,
    )

    return fig

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
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

    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    df = df.drop_duplicates()
    return df.reset_index(drop=True)

def candidate_accuracy_cols(df: pd.DataFrame) -> List[str]:
    cands = [c for c in df.columns if re.search(r"(acc|accuracy)", c, re.I)]
    if "ACC_val" in cands:
        cands.remove("ACC_val")
        cands = ["ACC_val"] + cands
    return cands or [c for c in numeric_cols(df) if "loss" not in c.lower()][:5]

def candidate_cost_cols(df: pd.DataFrame) -> List[str]:
    keys = ["emission", "co2", "energy", "time", "latency", "cost", "power"]
    cands = [c for c in df.columns if any(k in c.lower() for k in keys)]
    order = ["emissions", "emissions_gCO2eq", "energy_consumed", "energy_J", "train_epoch_time_ms"]
    ranked = [c for c in order if c in df.columns]
    ranked += [c for c in cands if c not in ranked]
    return ranked or numeric_cols(df)

def _safe_series(df: pd.DataFrame, col_name: str) -> pd.Series:
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

def suggest_parameter_ranges(df: pd.DataFrame, good_mask: np.ndarray, pcols: List[str]):
    """
    From the 'good' runs (mask), suggest useful ranges / categories for each param.
    Returns dict[param] -> (low, high) for numeric or ('categorical', [values]) for categorical.
    """
    recs: Dict[str, Tuple] = {}
    good = df.loc[good_mask].copy()
    if good.empty:
        return recs

    for p in pcols:
        if p not in df.columns:
            continue

        col_all = df[p]
        col_good = good[p].dropna()
        if col_good.empty:
            continue

        if pd.api.types.is_numeric_dtype(col_all):
            low, high = float(col_good.min()), float(col_good.max())
            if low == high:
                continue
            recs[p] = (low, high)
        else:
            counts = col_good.value_counts()
            # take up to 5 most frequent categories
            top_vals = list(counts.index[:5])
            if not top_vals:
                continue
            recs[p] = ("categorical", top_vals)

    return recs


def neighbor_grid(val, scale: float = 0.5):
    """
    Small grid of values around a given numeric value.
    For ints: +/- 1; for floats: +/- scale * |val| (fallback to 0.1 if val≈0).
    """
    if isinstance(val, (int, np.integer)):
        base = int(val)
        grid = [base - 1, base, base + 1]
        return [g for g in grid if g >= 0]

    try:
        v = float(val)
    except Exception:
        return [val]

    step = abs(v) * scale if abs(v) > 1e-8 else 0.1
    grid = [v - step, v, v + step]
    return [float(f"{g:.6g}") for g in grid]

def _scalar_from_row(row: pd.Series, key: Optional[str]):
    if not key or key not in row.index:
        return None
    v = row[key]
    return v.iloc[0] if isinstance(v, pd.Series) else v

def make_tooltip_text(row: pd.Series, id_col: Optional[str], param_columns: List[str], 
                     acc_col: str, cost_col: str) -> str:
    rid_val = _scalar_from_row(row, id_col)
    rid = str(rid_val) if rid_val is not None else f"(idx {row.name})"
    params = ", ".join([f"{p}={_scalar_from_row(row, p)}" for p in param_columns if p in row.index])
    a = _scalar_from_row(row, acc_col); c = _scalar_from_row(row, cost_col)
    a_txt = f"{a:.4g}" if (a is not None and pd.notnull(a) and np.isfinite(float(a))) else str(a)
    c_txt = f"{c:.4g}" if (c is not None and pd.notnull(c) and np.isfinite(float(c))) else str(c)
    return f"id={rid}<br>{acc_col}={a_txt}<br>{cost_col}={c_txt}<br>{params}"

# ----------------------------- NEW: Advanced Visualization Functions -----------------------------

def create_pairwise_analysis(df: pd.DataFrame, pcols: List[str], acc_col: str, cost_col: str):
    """Create pairwise scatter plots for key hyperparameter combinations."""
    if not pcols or len(pcols) < 2:
        return None
    
    # Select numeric parameters with variation
    numeric_params = [p for p in pcols if pd.api.types.is_numeric_dtype(df[p]) and df[p].var() > 0]
    
    # Add categorical params too
    categorical_params = [p for p in pcols if not pd.api.types.is_numeric_dtype(df[p]) and df[p].nunique() > 1]
    
    all_params = numeric_params + categorical_params
    
    if len(all_params) < 2:
        return None
    
    # Limit to top parameters by importance (variance for numeric, count for categorical)
    top_params = all_params[:min(4, len(all_params))]
    
    # Create pairs
    pairs = list(combinations(top_params, 2))[:6]  # Max 6 pairs
    
    if not pairs:
        return None
    
    n_pairs = len(pairs)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{x} vs {y}" for x, y in pairs],
        vertical_spacing=0.15,  # Increased spacing
        horizontal_spacing=0.12  # Increased spacing
    )
    
    for idx, (x_param, y_param) in enumerate(pairs):
        row_idx = idx // cols + 1
        col_idx = idx % cols + 1
        
        # Handle categorical variables
        df_plot = df.copy()
        x_is_cat = not pd.api.types.is_numeric_dtype(df[x_param])
        y_is_cat = not pd.api.types.is_numeric_dtype(df[y_param])
        
        if x_is_cat:
            cat_map = {cat: i for i, cat in enumerate(sorted(df[x_param].unique()))}
            x_vals = df[x_param].map(cat_map)
        else:
            x_vals = df[x_param]
        
        if y_is_cat:
            cat_map = {cat: i for i, cat in enumerate(sorted(df[y_param].unique()))}
            y_vals = df[y_param].map(cat_map)
        else:
            y_vals = df[y_param]
        
        hover_text = df.apply(
            lambda r: f"{x_param}={r[x_param]}<br>{y_param}={r[y_param]}<br>{acc_col}={r[acc_col]:.4f}",
            axis=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    color=df[acc_col],
                    colorscale='RdYlGn',
                    size=8,
                    showscale=(idx == 0),
                    colorbar=dict(title=acc_col, x=1.02) if idx == 0 else None,
                    line=dict(width=0.5, color='black')
                ),
                text=hover_text,
                hoverinfo='text',
                showlegend=False
            ),
            row=row_idx, col=col_idx
        )
        
        # Update axes labels with better formatting to avoid overlap
        fig.update_xaxes(
            title_text=x_param, 
            row=row_idx, col=col_idx,
            tickangle=-45,  # Rotate labels
            title_font=dict(size=10),
            tickfont=dict(size=9)
        )
        fig.update_yaxes(
            title_text=y_param, 
            row=row_idx, col=col_idx,
            title_font=dict(size=10),
            tickfont=dict(size=9)
        )
    
    fig.update_layout(
        title_text="Pairwise Hyperparameter Analysis (2D Slices of Solution Space)",
        height=350 * rows,  # Increased height
        showlegend=False,
        margin=dict(t=80, b=80)  # Add margins
    )
    
    return fig


def create_heatmap(
    df: pd.DataFrame,
    pcols: List[str],
    acc_col: str,
    x_param: str,
    y_param: str,
):
    """
    Create ONE big performance heatmap for a chosen pair of numeric parameters.
    Cells are equally sized (categorical axes).
    """
    # Safety: both params must exist and be numeric with variation
    for p in (x_param, y_param):
        if p not in df.columns:
            return None
        if (not pd.api.types.is_numeric_dtype(df[p])) or df[p].nunique() < 2:
            return None

    # Pivot table: y on rows, x on columns
    pivot = df.pivot_table(
        values=acc_col,
        index=y_param,
        columns=x_param,
        aggfunc="mean",
    )

    if pivot.empty:
        return None

    # Build hover text
    hover_text = []
    for i, y_val in enumerate(pivot.index):
        row_text = []
        for j, x_val in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            row_text.append(
                f"{x_param}: {x_val:.4g}<br>"
                f"{y_param}: {y_val:.4g}<br>"
                f"{acc_col}: {val:.4f}"
            )
        hover_text.append(row_text)

    # Use string labels so each cell has the same size
    x_labels = [f"{v:.4g}" for v in pivot.columns]
    y_labels = [f"{v:.4g}" for v in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn",
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            hovertext=hover_text,
            hoverinfo="text",
            showscale=True,
            colorbar=dict(
                title=acc_col,
                len=0.8,
            ),
            xgap=2,
            ygap=2,
        )
    )

    # Axes as categorical to force equal partitions
    fig.update_xaxes(
        title_text=x_param,
        tickangle=-45,
        title_font=dict(size=12),
        tickfont=dict(size=11),
        type="category",
        categoryorder="array",
        categoryarray=x_labels,
    )

    fig.update_yaxes(
        title_text=y_param,
        title_font=dict(size=12),
        tickfont=dict(size=11),
        type="category",
        categoryorder="array",
        categoryarray=y_labels,
    )

    fig.update_layout(
        title_text=f"{acc_col} Heatmap: {y_param} vs {x_param}",
        height=500,
        margin=dict(t=80, b=80, l=80, r=80),
        showlegend=False,
    )

    return fig




def create_main_effects(df: pd.DataFrame, pcols: List[str], acc_col: str):
    """Main effects plot that fits on screen nicely."""
    if not pcols:
        return None
    
    varying_params = [p for p in pcols if df[p].nunique() > 1]
    if not varying_params:
        return None
    
    top_params = varying_params[:min(5, len(varying_params))]
    
    fig = make_subplots(
        rows=1, cols=len(top_params),
        subplot_titles=[f"{p}" for p in top_params],
        horizontal_spacing=0.12,
    )
    
    y_min = df[acc_col].min()
    y_max = df[acc_col].max()
    y_range = y_max - y_min
    y_axis_min = max(0, y_min - 0.1 * y_range)
    y_axis_max = y_max + 0.2 * y_range
    
    for idx, param in enumerate(top_params, 1):
        if pd.api.types.is_numeric_dtype(df[param]):
            grouped = df.groupby(param)[acc_col].agg(['mean', 'std', 'count']).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=grouped[param],
                    y=grouped['mean'],
                    mode='lines+markers',
                    name=param,
                    error_y=dict(type='data', array=grouped['std']),
                    line=dict(width=2),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=1, col=idx
            )
        else:
            grouped = df.groupby(param)[acc_col].mean().reset_index()
            text_vals = [f'{v:.3f}' for v in grouped[acc_col].values]
            
            fig.add_trace(
                go.Bar(
                    x=grouped[param].astype(str),
                    y=grouped[acc_col],
                    name=param,
                    text=text_vals,
                    textposition='outside',
                    textfont=dict(size=11, color='black'),
                    showlegend=False,
                    marker=dict(
                        color=grouped[acc_col],
                        colorscale='RdYlGn',
                        line=dict(color='black', width=1)
                    )
                ),
                row=1, col=idx
            )
        
        fig.update_xaxes(
            title_text=param,
            row=1, col=idx,
            tickangle=-45,
            title_font=dict(size=11),
            tickfont=dict(size=10),
        )
        
        fig.update_yaxes(
            title_text=acc_col if idx == 1 else "",
            row=1, col=idx,
            range=[y_axis_min, y_axis_max]
        )
    
    # Just set a reasonable height, let width be automatic
    fig.update_layout(
        title_text="Main Effects: Individual Parameter Impact",
        height=450,
        showlegend=False,
        margin=dict(b=120, t=80, l=80, r=40)
    )
    
    return fig



# ----------------------------- NEW: Smart Recommendations -----------------------------

def generate_smart_recommendations(df: pd.DataFrame, front: pd.DataFrame, 
                                   pcols: List[str], acc_col: str, cost_col: str,
                                   id_col: Optional[str], n_recs: int = 5) -> List[Dict]:
    """
    Generate intelligent recommendations for next experiments.
    Works with ANY CSV structure automatically.
    
    Strategies:
    1. EXPLOIT: Refine around best configuration
    2. EXPLORE: Test underexplored regions
    3. BALANCE: Optimize accuracy vs sustainability
    4. INTERPOLATE: Fill gaps in tested values
    """
    recommendations = []
    
    if len(df) == 0:
        return recommendations
    
    # Get best configuration
    best_idx = df[acc_col].idxmax()
    best_config = df.loc[best_idx]
    
    # Get top performers for analysis
    top_n = min(5, len(df))
    top_df = df.nlargest(top_n, acc_col)
    
    # Strategy 1: EXPLOIT - Refine around best
    rec1 = {
        'strategy': '🎯 EXPLOIT',
        'rationale': 'Refine around best known configuration',
        'priority': 'HIGH',
        'config': {},
        'details': 'Small variations of optimal parameters'
    }
    
    for p in pcols:
        if p in best_config.index and pd.notnull(best_config[p]):
            val = best_config[p]
            if pd.api.types.is_numeric_dtype(df[p]):
                rec1['config'][p] = f"{float(val):.6g}"
            else:
                rec1['config'][p] = str(val)
    
    if rec1['config']:
        rec1['expected_performance'] = f"{acc_col} ≈ {best_config[acc_col]:.4f} ± 0.01"
        if cost_col in best_config.index and pd.notnull(best_config[cost_col]):
            rec1['expected_cost'] = f"{cost_col} ≈ {best_config[cost_col]:.4f}"
        recommendations.append(rec1)
    
    # Strategy 2: EXPLORE - Underexplored regions
    for p in pcols:
        if len(recommendations) >= n_recs:
            break
        
        if p not in df.columns:
            continue
        
        if pd.api.types.is_numeric_dtype(df[p]):
            # For numeric: find underexplored values
            all_vals = sorted(df[p].dropna().unique())
            if len(top_df) > 0:
                top_vals = set(top_df[p].dropna().unique())
                unexplored = [v for v in all_vals if v not in top_vals]
                
                if unexplored and len(unexplored) < len(all_vals):
                    rec = {
                        'strategy': '🔍 EXPLORE',
                        'rationale': f'Test underexplored {p} value',
                        'priority': 'MEDIUM',
                        'config': {},
                        'details': f'Value rarely tested in top performers'
                    }
                    
                    explore_val = unexplored[len(unexplored)//2]
                    rec['config'][p] = f"{float(explore_val):.6g}"
                    
                    # Fill other params from best config
                    for other_p in pcols:
                        if other_p != p and other_p in best_config.index and pd.notnull(best_config[other_p]):
                            val = best_config[other_p]
                            if pd.api.types.is_numeric_dtype(df[other_p]):
                                rec['config'][other_p] = f"{float(val):.6g}"
                            else:
                                rec['config'][other_p] = str(val)
                    
                    rec['expected_performance'] = f"{acc_col}: 0.60 - 0.75 (uncertain)"
                    recommendations.append(rec)
        else:
            # For categorical: find underrepresented categories
            all_cats = set(df[p].dropna().unique())
            if len(top_df) > 0:
                top_cats = set(top_df[p].dropna().unique())
                underexplored = all_cats - top_cats
                
                if underexplored and len(recommendations) < n_recs:
                    rec = {
                        'strategy': '🔍 EXPLORE',
                        'rationale': f'Test underexplored {p} category: {list(underexplored)[0]}',
                        'priority': 'MEDIUM',
                        'config': {},
                        'details': 'Category absent from top performers'
                    }
                    
                    explore_cat = list(underexplored)[0]
                    rec['config'][p] = str(explore_cat)
                    
                    # Fill other params from best config
                    for other_p in pcols:
                        if other_p != p and other_p in best_config.index and pd.notnull(best_config[other_p]):
                            val = best_config[other_p]
                            if pd.api.types.is_numeric_dtype(df[other_p]):
                                rec['config'][other_p] = f"{float(val):.6g}"
                            else:
                                rec['config'][other_p] = str(val)
                    
                    rec['expected_performance'] = f"{acc_col}: uncertain"
                    recommendations.append(rec)
    
    # Strategy 3: BALANCE - Accuracy vs Cost trade-off
    if cost_col in df.columns and len(df) > 5 and len(recommendations) < n_recs:
        df_sorted = df.sort_values([acc_col, cost_col], ascending=[False, True])
        efficient_idx = min(2, len(df_sorted) - 1)
        efficient = df_sorted.iloc[efficient_idx]
        
        # Only add if meaningfully different from best
        if efficient.name != best_idx:
            rec = {
                'strategy': '⚖️ BALANCE',
                'rationale': 'Optimize accuracy/sustainability trade-off',
                'priority': 'MEDIUM',
                'config': {},
                'details': 'Good performance with lower environmental impact'
            }
            
            for p in pcols:
                if p in efficient.index and pd.notnull(efficient[p]):
                    val = efficient[p]
                    if pd.api.types.is_numeric_dtype(df[p]):
                        rec['config'][p] = f"{float(val):.6g}"
                    else:
                        rec['config'][p] = str(val)
            
            if rec['config']:
                rec['expected_performance'] = f"{acc_col} ≈ {efficient[acc_col]:.4f}"
                if pd.notnull(efficient[cost_col]):
                    rec['expected_cost'] = f"{cost_col} ≈ {efficient[cost_col]:.4f}"
                    # Calculate efficiency ratio
                    efficiency = efficient[acc_col] / (efficient[cost_col] + 1e-6)
                    rec['details'] = f"Efficiency ratio: {efficiency:.2f} (higher is better)"
                recommendations.append(rec)
    
    # Strategy 4: INTERPOLATE - Fill gaps
    if len(recommendations) < n_recs:
        numeric_params = [p for p in pcols if pd.api.types.is_numeric_dtype(df[p]) 
                         and df[p].nunique() >= 2]
        
        for p in numeric_params:
            if len(recommendations) >= n_recs:
                break
            
            vals = sorted(df[p].dropna().unique())
            if len(vals) >= 2:
                # Find largest gap
                gaps = [(vals[i+1], vals[i], vals[i+1] - vals[i]) for i in range(len(vals)-1)]
                largest_gap = max(gaps, key=lambda x: x[2])
                
                # Only suggest if gap is significant
                if largest_gap[2] > (vals[-1] - vals[0]) / (len(vals) * 2):
                    midpoint = (largest_gap[0] + largest_gap[1]) / 2
                    
                    rec = {
                        'strategy': '🔗 INTERPOLATE',
                        'rationale': f'Fill gap in {p} values',
                        'priority': 'LOW',
                        'config': {},
                        'details': f'Test intermediate value between {largest_gap[1]:.4g} and {largest_gap[0]:.4g}'
                    }
                    
                    rec['config'][p] = f"{midpoint:.6g}"
                    
                    # Fill other params from best config
                    for other_p in pcols:
                        if other_p != p and other_p in best_config.index and pd.notnull(best_config[other_p]):
                            val = best_config[other_p]
                            if pd.api.types.is_numeric_dtype(df[other_p]):
                                rec['config'][other_p] = f"{float(val):.6g}"
                            else:
                                rec['config'][other_p] = str(val)
                    
                    # Estimate performance
                    nearby = df[df[p].between(largest_gap[1], largest_gap[0])]
                    if len(nearby) > 0:
                        nearby_perf = nearby[acc_col].mean()
                        rec['expected_performance'] = f"{acc_col} ≈ {nearby_perf:.4f} (interpolated)"
                    
                    recommendations.append(rec)
    
    return recommendations[:n_recs]


# ----------------------------- NEW: SHAP Analysis for Clustering -----------------------------

def prepare_features_for_shap(df: pd.DataFrame, pcols: List[str]) -> Tuple[np.ndarray, List[str], ColumnTransformer]:
    """Prepare features for SHAP analysis with proper encoding."""
    numeric_features = [p for p in pcols if pd.api.types.is_numeric_dtype(df[p])]
    categorical_features = [p for p in pcols if p not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ] if categorical_features else [
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    X = preprocessor.fit_transform(df[pcols])
    
    # Get feature names after encoding
    feature_names = []
    if categorical_features:
        try:
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_names = []
            for i, feat in enumerate(categorical_features):
                cats = cat_encoder.categories_[i][1:]  # Skip first (dropped)
                cat_names.extend([f"{feat}_{cat}" for cat in cats])
            feature_names = numeric_features + cat_names
        except:
            feature_names = numeric_features
    else:
        feature_names = numeric_features
    
    return X, feature_names, preprocessor


def compute_shap_for_clusters(df: pd.DataFrame, pcols: List[str], cluster_labels: np.ndarray, 
                               model_type: str = "RandomForest") -> Tuple[Optional[shap.Explanation], Optional[object]]:
    """Compute SHAP values for cluster assignments using a surrogate model."""
    
    if not SHAP_OK or not SKLEARN_OK:
        return None, None
    
    try:
        # Prepare features
        X, feature_names, preprocessor = prepare_features_for_shap(df, pcols)
        
        # Filter out noise points if using DBSCAN
        valid_mask = cluster_labels != -1
        if not valid_mask.any():
            return None, None
        
        X_valid = X[valid_mask]
        y_valid = cluster_labels[valid_mask]
        
        # Check if we have enough samples
        if len(X_valid) < 2:
            return None, None
        
        # Train surrogate model based on user selection
        if model_type == "RandomForest":
            surrogate = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "LogisticRegression":
            surrogate = LogisticRegression(max_iter=1000, random_state=42)
        else:
            # Default to RandomForest
            surrogate = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        surrogate.fit(X_valid, y_valid)
        
        # Compute SHAP values
        if model_type == "RandomForest":
            explainer = shap.TreeExplainer(surrogate)
            shap_values = explainer.shap_values(X_valid)
        else:
            # For linear models, use LinearExplainer
            explainer = shap.LinearExplainer(surrogate, X_valid)
            shap_values = explainer.shap_values(X_valid)
        
        # Handle different SHAP value formats
        # For multi-class: shap_values is a list of arrays, one per class
        # For binary: shap_values is a single array
        
        if isinstance(shap_values, list):
            # Multi-class: stack into 3D array (samples, features, classes)
            shap_values_array = np.stack(shap_values, axis=2)
        else:
            # Binary or single output: keep as 2D array
            shap_values_array = shap_values
        
        # Get expected values
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) == 1:
                expected_value = explainer.expected_value[0]
            else:
                expected_value = explainer.expected_value
        else:
            expected_value = explainer.expected_value
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values_array,
            base_values=expected_value,
            data=X_valid,
            feature_names=feature_names
        )
        
        return explanation, surrogate
        
    except Exception as e:
        st.warning(f"Could not compute SHAP values: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


def plot_shap_importance(shap_explanation: shap.Explanation, max_features: int = 10):
    """Plot SHAP feature importance."""
    if shap_explanation is None:
        return None
    
    try:
        # Handle both 2D and 3D SHAP values arrays
        shap_vals = shap_explanation.values
        
        # If 3D (multi-class), average across classes first
        if len(shap_vals.shape) == 3:
            shap_vals = np.abs(shap_vals).mean(axis=2).mean(axis=0)
        # If 2D (binary or single output), average across samples
        elif len(shap_vals.shape) == 2:
            shap_vals = np.abs(shap_vals).mean(axis=0)
        else:
            # Fallback: just take absolute values
            shap_vals = np.abs(shap_vals)
        
        # Ensure we have feature names
        feature_names = shap_explanation.feature_names
        if feature_names is None or len(feature_names) != len(shap_vals):
            feature_names = [f"Feature_{i}" for i in range(len(shap_vals))]
        
        # Sort by importance
        idx = np.argsort(shap_vals)[::-1][:max_features]
        
        # Get top features
        top_vals = shap_vals[idx]
        top_names = [feature_names[i] for i in idx]
        
        fig = go.Figure(go.Bar(
            x=top_vals,
            y=top_names,
            orientation='h',
            marker=dict(
                color=top_vals, 
                colorscale='Viridis',
                line=dict(color='black', width=1)
            ),
            text=[f'{v:.4f}' for v in top_vals],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="SHAP Feature Importance for Cluster Assignment",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            height=400,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=150)  # More space for feature names
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not plot SHAP importance: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ----------------------------- Clustering Functions -----------------------------

def run_clustering(df_in: pd.DataFrame, x_col: str, y_col: str, method: str, params: dict):
    """Cluster on two selected metrics with different algorithms."""
    df_used = df_in.dropna(subset=[x_col, y_col]).copy()
    if df_used.empty:
        return None, None, df_used

    X = df_used[[x_col, y_col]].to_numpy()

    scale_needed = method in {"DBSCAN", "Agglomerative", "Spectral"}
    if scale_needed:
        try:
            X = StandardScaler().fit_transform(X)
        except Exception:
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

    elif method == "Agglomerative":
        k = int(params.get("k", 3))
        linkage = params.get("linkage", "ward")
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)

    elif method == "Spectral":
        k = int(params.get("k", 3))
        assign = params.get("assign_labels", "kmeans")
        model = SpectralClustering(n_clusters=k, assign_labels=assign, random_state=42, affinity="rbf")
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, model, df_used

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

# ----------------------------- UI Setup -----------------------------

st.set_page_config(page_title="yProv4ML Decision Support v2", layout="wide")

# Add logo and title
col_logo, col_title = st.columns([1, 5])
with col_logo:
    try:
        # Try to load logo from multiple possible locations
        # Use raw strings (r"") or forward slashes for Windows paths
        logo_paths = [
            r"C:\Users\admin\Pictures\Screenshots\yprov4ml.png",  # Windows path (raw string)
            "C:/Users/admin/Pictures/Screenshots/yprov4ml.png",  # Windows path (forward slashes)
            "/mnt/user-data/uploads/Screenshot_2025-11-03_194540.png",
            "/home/claude/yprov4ml_logo.png",
            "/mnt/user-data/outputs/yprov4ml_logo.png",
            "./yprov4ml_logo.png",  # Current directory
            "./yprov4ml.png",  # Alternative name
        ]
        logo_loaded = False
        for logo_path in logo_paths:
            try:
                st.image(logo_path, width=120)
                logo_loaded = True
                break
            except:
                continue
        if not logo_loaded:
            st.markdown("### 🧬")
    except:
        st.markdown("### 🧬")

with col_title:
    st.title("yProv4ML • Enhanced Decision-Support Dashboard v2")
    st.write("Upload a CSV to analyze **multi-dimensional solution space** with intelligent recommendations and SHAP analysis.")

st.markdown("---")

# ----------------------------- Sidebar: Upload CSV -----------------------------

st.sidebar.header("📂 1) Upload Data")
up = st.sidebar.file_uploader("Select a CSV file", type=["csv"], key="single_csv")

df = None
if up is not None:
    try:
        raw = load_csv_from_bytes(up.getvalue())
        raw = ensure_unique_columns(raw)
        df = clean_rows(raw)
        st.sidebar.success(f"✅ Loaded: {up.name}\n({len(df)} experiments)")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

if df is None:
    st.info("👈 Please upload a CSV file to begin analysis.")
    st.stop()

# ----------------------------- Structure Detection -----------------------------

id_col = detect_id_column(df)
pcols = param_cols(df)
ncols = numeric_cols(df)

# Filter out ID-like columns
ID_EXACT = {"id", "run_id", "run", "exp", "experiment"}
def is_id_like(name: str) -> bool:
    n = str(name).strip().lower()
    return (n in ID_EXACT) or n.endswith("_id")

def drop_id_like(cols):
    return [c for c in cols if not is_id_like(c)]

acc_cands = drop_id_like(candidate_accuracy_cols(df))
cost_cands = drop_id_like(candidate_cost_cols(df))
ncols_no_id = drop_id_like(ncols)

neutral = [c for c in ncols_no_id
           if c not in set(acc_cands) | set(cost_cands)
           and not str(c).startswith("param_")]

acc_options = list(dict.fromkeys(acc_cands + neutral))
cost_options = list(dict.fromkeys(cost_cands + neutral))

# ----------------------------- Sidebar: Metric Selection -----------------------------

st.sidebar.header("📊 2) Select Metrics")
acc_col = st.sidebar.selectbox(
    "Performance metric (maximize)",
    options=(acc_options or ncols_no_id), index=0
)
cost_col = st.sidebar.selectbox(
    "Cost/Sustainability metric (minimize)",
    options=(cost_options or ncols_no_id), index=0
)

# ----------------------------- Filter Data -----------------------------

work = df.dropna(subset=[acc_col, cost_col]).copy()
if len(work) < 2:
    st.warning("⚠️ Not enough rows with both selected metrics.")
    st.stop()

# ----------------------------- Main Dashboard -----------------------------

# Overview metrics
st.subheader("📈 Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Experiments", len(df))
with col2:
    st.metric("Parameters Detected", len(pcols))
with col3:
    st.metric(f"Mean {acc_col}", f"{df[acc_col].mean():.4f}")
with col4:
    st.metric(f"Mean {cost_col}", f"{df[cost_col].mean():.4f}")
# ----------------------------- Pareto data (for multiple sections) -----------------------------

cols_for_front = [acc_col, cost_col] + pcols

# Add id_col if present
if id_col:
    cols_for_front.append(id_col)

# Keep only existing columns
cols_for_front = [c for c in cols_for_front if c in work.columns]

# ✅ IMPORTANT: remove duplicates while preserving order
cols_for_front = list(dict.fromkeys(cols_for_front))

front = nondominated_front(work[cols_for_front], acc_col, cost_col)


# ----------------------------- NEW: Advanced Visualizations -----------------------------

st.subheader("🔬 Solution Space Analysis")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(
    ["Pairwise Analysis", "Heatmaps", "Main Effects", "Correlations"]
)


with tab1:
    st.markdown("**2D slices of your multi-dimensional solution space**")
    fig_pairwise = create_pairwise_analysis(work, pcols, acc_col, cost_col)
    if fig_pairwise:
        st.plotly_chart(fig_pairwise, use_container_width=True)
    else:
        st.info("Need at least 2 parameters to show pairwise analysis.")

with tab2:
    st.markdown("**Performance heatmap across parameter combinations**")

    # numeric hyperparameters we can use for axes
    numeric_params = [
        p for p in pcols
        if p in work.columns
        and pd.api.types.is_numeric_dtype(work[p])
        and work[p].nunique() >= 2
    ]

    if len(numeric_params) < 2:
        st.info("Need at least 2 numeric parameters to show a heatmap.")
    else:
        col_x, col_y = st.columns(2)
        with col_x:
            x_param = st.selectbox(
                "X-axis parameter",
                options=numeric_params,
                index=0,
            )
        with col_y:
            # Y options exclude currently selected X
            y_options = [p for p in numeric_params if p != x_param]
            y_param = st.selectbox(
                "Y-axis parameter",
                options=y_options,
                index=0,
            )

        fig_heatmap = create_heatmap(work, pcols, acc_col, x_param, y_param)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Could not build heatmap for this parameter pair.")



with tab3:
    st.markdown("**Individual parameter impact on performance**")
    fig_main = create_main_effects(work, pcols, acc_col)
    if fig_main:
        st.plotly_chart(fig_main, use_container_width=True) 
    else:
        st.info("Need parameters with variation to show main effects.")

with tab4:
    st.markdown("**Correlation matrix between hyperparameters and selected metrics.**")

    fig_corr = create_correlation_heatmap(work, pcols, acc_col, cost_col)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info(
            "Need at least 2 numeric hyperparameters (plus metrics) to compute correlations."
        )


# ----------------------------- Pareto Analysis -----------------------------

st.subheader("🎯 Pareto Frontier")

st.write(
    f"Found **{len(front)}** Pareto-optimal configurations "
    f"(maximize `{acc_col}`, minimize `{cost_col}`)."
)


# Pareto plot
def plot_pareto(df_all, df_front, acc_col, cost_col, id_col, pcols):
    fig = go.Figure()
    
    # Use safe 1D series in case of weird column duplications
    x_all = _safe_series(df_all, cost_col)
    y_all = _safe_series(df_all, acc_col)
    x_front = _safe_series(df_front, cost_col)
    y_front = _safe_series(df_front, acc_col)
    
    hover_all = df_all.apply(
        lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col),
        axis=1,
    )
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        mode="markers",
        name="All experiments",
        opacity=0.4,
        marker=dict(size=8, color="lightblue", line=dict(width=0.5, color="gray")),
        text=hover_all,
        hoverinfo="text",
    ))
    
    hover_front = df_front.apply(
        lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col),
        axis=1,
    )
    fig.add_trace(go.Scatter(
        x=x_front,
        y=y_front,
        mode="markers",
        name="Pareto frontier",
        marker=dict(
            size=14,
            color="red",
            symbol="diamond",
            line=dict(width=2, color="darkred"),
        ),
        hovertext=hover_front,
        hoverinfo="text",
    ))
    
    fig.update_layout(
        title="Performance vs Sustainability Trade-off",
        xaxis_title=f"{cost_col} (minimize)",
        yaxis_title=f"{acc_col} (maximize)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


st.plotly_chart(plot_pareto(work, front, acc_col, cost_col, id_col, pcols), use_container_width=True)

# ----------------------------- Clustering with SHAP (Optional) -----------------------------

with st.expander("🎲 Advanced: Clustering Analysis with SHAP Importance"):
    st.sidebar.subheader("⚙️ Clustering Settings")
    clustering_mode = st.sidebar.radio("Cluster on:", ["Full dataset", "Pareto-only"])
    method = st.sidebar.selectbox("Method", ["KMeans", "DBSCAN", "Agglomerative", "Spectral"], index=0)
    
    if SHAP_OK and SKLEARN_OK:
        st.sidebar.subheader("🔍 SHAP Settings")
        surrogate_model = st.sidebar.selectbox(
            "Surrogate Model for SHAP",
            ["RandomForest", "LogisticRegression"],
            index=0,
            help="Choose a model to explain cluster assignments. RandomForest recommended for complex patterns."
        )
    
    if method == "KMeans":
        k = st.sidebar.slider("K (clusters)", 2, 12, 3)
        params = {"k": k}
    elif method == "DBSCAN":
        eps = st.sidebar.slider("eps", 0.05, 5.0, 0.5, 0.05)
        min_samples = st.sidebar.slider("min_samples", 2, 50, 5)
        params = {"eps": eps, "min_samples": min_samples}
    elif method == "Agglomerative":
        k = st.sidebar.slider("K (clusters)", 2, 12, 3)
        linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])
        params = {"k": k, "linkage": linkage}
    else:  # Spectral
        k = st.sidebar.slider("K (clusters)", 2, 12, 3)
        params = {"k": k}
    
    cluster_df = front if clustering_mode == "Pareto-only" else work
    
    if SKLEARN_OK and len(cluster_df) >= 2:
        labels, model, used = run_clustering(cluster_df, cost_col, acc_col, method, params)
        
        if labels is not None:
            used = used.copy()
            used["cluster"] = labels
            used["cluster_label"] = used["cluster"].apply(lambda z: "noise" if z == -1 else str(z))
            
            col_plot, col_shap = st.columns([1, 1])
            
            with col_plot:
                figc = px.scatter(
                    used, x=cost_col, y=acc_col, color="cluster_label",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title=f"{method} Clustering on {clustering_mode}",
                    hover_data=[c for c in pcols if c in used.columns]
                )
                st.plotly_chart(figc, use_container_width=True)
            
            with col_shap:
                if SHAP_OK and len(pcols) > 0:
                    st.markdown("**SHAP Feature Importance**")
                    with st.spinner("Computing SHAP values..."):
                        shap_exp, surrogate = compute_shap_for_clusters(
                            cluster_df, pcols, labels, surrogate_model
                        )
                        
                        if shap_exp is not None:
                            fig_shap = plot_shap_importance(shap_exp, max_features=10)
                            if fig_shap:
                                st.plotly_chart(fig_shap, use_container_width=True)
                                
                                st.info(f"✅ Surrogate model ({surrogate_model}) accuracy: "
                                       f"{surrogate.score(shap_exp.data, labels[labels != -1]):.3f}")
                        else:
                            st.warning("Could not compute SHAP values")
                else:
                    st.info("Install SHAP for feature importance: `pip install shap`")
            
            summary = summarize_clusters(used, "cluster", [acc_col, cost_col])
            if not summary.empty:
                st.dataframe(summary, use_container_width=True)
    else:
        st.info("Install scikit-learn for clustering: `pip install scikit-learn`")

# ----------------------------- Prescriptive analytics (simple) -----------------------------
with st.expander("Prescriptive analytics"):

    q_acc = st.slider(
        "Top-quantile for accuracy (good ≥)",
        0.50, 0.95, 0.80, 0.05,
        key="presc_q_acc",
    )
    q_cost = st.slider(
        "Bottom-quantile for cost (good ≤)",
        0.05, 0.50, 0.25, 0.05,
        key="presc_q_cost",
    )

    thr_acc = work[acc_col].quantile(q_acc)
    thr_cost = work[cost_col].quantile(q_cost)

    st.write(
        f"Thresholds → **{acc_col} ≥ {thr_acc:.4g}** "
        f"and **{cost_col} ≤ {thr_cost:.4g}**."
    )

    good_mask = (work[acc_col] >= thr_acc) & (work[cost_col] <= thr_cost)
    st.write(
        f"Selected **{int(good_mask.sum())}/{len(work)}** runs as *high-utility* candidates."
    )

    if good_mask.any():
        cols_view = [acc_col, cost_col] + ([id_col] if id_col else []) + pcols
        cols_view = [c for c in cols_view if c in work.columns]
        cols_view = list(dict.fromkeys(cols_view))
        good = work.loc[good_mask, cols_view].copy()
        st.dataframe(
            good.sort_values([cost_col, acc_col], ascending=[True, False]).head(20),
            use_container_width=True,
        )
    else:
        st.info(
            "No runs meet both thresholds. Try adjusting the sliders or switching cost metric."
        )


# ----------------------------- NEW: Smart Recommendations -----------------------------

st.subheader("💡 Intelligent Recommendations")
st.markdown("**Next experiments to try (inference within the same solution space)**")

recommendations = generate_smart_recommendations(
    df, front, pcols, acc_col, cost_col, id_col, n_recs=5
)

if recommendations:
    for idx, rec in enumerate(recommendations, 1):
        with st.expander(f"**Recommendation {idx}: {rec['strategy']}** - {rec['rationale']}", expanded=(idx==1)):
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**Priority:** {rec['priority']}")
                st.markdown(f"**Details:** {rec.get('details', 'N/A')}")
                
                if 'expected_performance' in rec:
                    st.markdown(f"**Expected Performance:** {rec['expected_performance']}")
                
                if 'expected_cost' in rec:
                    st.markdown(f"**Expected Cost:** {rec['expected_cost']}")
                
                st.markdown("**Configuration:**")
                st.json(rec['config'])
            
            with col_b:
                # Show how this compares to current best
                if 'expected_performance' in rec:
                    best_perf = df[acc_col].max()
                    st.metric("vs Best", 
                             f"{best_perf:.4f}",
                             delta=None)
else:
    st.info("No recommendations generated. Try uploading more data.")

# ----------------------------- Download Recommendations -----------------------------

if recommendations:
    st.download_button(
        label="📥 Download Recommendations (JSON)",
        data=json.dumps(recommendations, indent=2),
        file_name="yprov_recommendations.json",
        mime="application/json"
    )

# ----------------------------- Footer -----------------------------

st.markdown("---")
st.caption("🧬 Built for provenance-guided hyperparameter optimization with yProv4ML")
st.caption("📊 Upload any CSV with hyperparameters and metrics to get intelligent recommendations")
st.caption("🆕 v2: Fixed overlapping labels, added SHAP importance with surrogate model selection")