# src/prov_to_csv.py
import argparse
import glob
import re
from pathlib import Path
import pandas as pd
from prov.model import ProvDocument, QualifiedName, ProvActivity, ProvEntity

BASE_ROOT = Path("data/prov")
OUT_DIR = Path("data/unified")
DEBUG = False  # set via --debug

# -------------------------- utils --------------------------

def dbg(msg: str):
    if DEBUG:
        print(msg)

def qn_to_prefixed(qn) -> str:
    if isinstance(qn, QualifiedName):
        pref = qn.namespace.prefix if qn.namespace else None
        local = qn.localpart
        return f"{pref}:{local}" if pref else local
    return str(qn)

def normalize_key(prefixed: str) -> str:
    if prefixed.startswith("metric:"):
        return prefixed.split(":", 1)[1]
    if prefixed.startswith("param:"):
        return "param_" + prefixed.split(":", 1)[1]
    return (
        prefixed.replace("context:", "context_")
                .replace("art:", "artifact_")
                .replace("user:", "user_")
    )

def _looks_like_index(series: pd.Series) -> bool:
    """Heuristic: monotonically non-decreasing small ints starting at ~0/1; or column named like step/epoch."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    # lots of zeros/ones or small ints?
    if (s >= 0).all() and (s <= 1e6).all():
        # monotonic-ish
        if s.is_monotonic_increasing or s.is_monotonic:
            return True
    return False

def _is_timestamp_value(x: float) -> bool:
    # reject obvious UNIX timestamps in seconds (>= 1e9 is already ms-ish), ms, ns (we can't reach those typical values in floats)
    try:
        return pd.notna(x) and (float(x) > 1e9)
    except Exception:
        return False

def _is_timestamp_colname(name: str) -> bool:
    n = str(name).lower()
    return ("timestamp" in n) or ("time" in n) or ("loggingitemkind.metric" in n) or n in {"ms","ns"}

def _choose_metric_column(df: pd.DataFrame, metric_name: str | None = None) -> str | None:
    """Choose the true value column, handling 3-col {step, value, timestamp} logs."""
    cols = list(df.columns)

    # 0) If exactly 3 columns and the RIGHTMOST looks like a timestamp -> pick the MIDDLE
    if len(cols) == 3:
        c0, c1, c2 = cols
        last_vals = pd.to_numeric(df[c2], errors="coerce").dropna()
        if _is_timestamp_colname(c2) or (not last_vals.empty and _is_timestamp_value(last_vals.iloc[-1])):
            if pd.api.types.is_numeric_dtype(df[c1]):
                return c1

    lower = {str(c).lower(): c for c in cols}

    # 1) Canonical value names
    for name in ("value", "metric_value", "val"):
        if name in lower and pd.api.types.is_numeric_dtype(df[lower[name]]):
            return lower[name]

    # 2) Two-column pattern: {index, value}
    if len(cols) == 2:
        nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(nums) == 2:
            a, b = nums
            if _looks_like_index(df[a]) and not _looks_like_index(df[b]): return b
            if _looks_like_index(df[b]) and not _looks_like_index(df[a]): return a

    # 3) General fallback: numeric, non-index, non-timestamp column (prefer rightmost)
    reject_exact = {"step", "iter", "iteration", "epoch", "index"}
    candidates = []
    for c in cols:
        lc = str(c).lower()
        if lc in reject_exact or _is_timestamp_colname(lc):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not _looks_like_index(df[c]):
            candidates.append(c)
    if candidates:
        return candidates[-1]

    return None

# ----------------------- metric reading -----------------------

def read_metric_csv(csv_path: Path, metric_name: str | None = None):
    """
    Read a metric CSV file and return the last numeric value, robust to shapes like:
    - {step, value}
    - {step, value, timestamp}
    - generic CSV with a 'value'/'val' column
    """
    try:
        if not csv_path.exists():
            dbg(f"   ❓ Missing metric CSV: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        if df.empty:
            dbg(f"   ⚠️ Empty metric CSV: {csv_path}")
            return None

        # Special-case: common 3-col header from your logs where col0 == metric name (step/index),
        # col1 == actual numeric value, col2 == timestamp (or 'LoggingItemKind.METRIC')
        if len(df.columns) == 3:
            c0, c1, c2 = df.columns
            last_vals = pd.to_numeric(df[c2], errors="coerce").dropna()
            if _is_timestamp_colname(c2) or (not last_vals.empty and _is_timestamp_value(last_vals.iloc[-1])):
                if pd.api.types.is_numeric_dtype(df[c1]):
                    val_series = pd.to_numeric(df[c1], errors="coerce").dropna()
                    if not val_series.empty:
                        val = float(val_series.iloc[-1])
                        dbg(f"   ✓ {csv_path.name}: 3-col pattern -> using '{c1}' -> last={val}")
                        # For non-time metrics, guard against accidentally picking timestamps
                        looks_timey_metric = metric_name and any(s in metric_name.lower() for s in ["time", "ms", "epoch_time"])
                        if not looks_timey_metric and _is_timestamp_value(val):
                            dbg(f"   🤨 middle '{c1}' in {csv_path.name} looks like a timestamp ({val}) — ignoring")
                            return None
                        return val

        # Otherwise, pick via heuristic chooser
        col = _choose_metric_column(df, metric_name)
        if col is None:
            dbg(f"   ⚠️ No usable value column in {csv_path}. Columns={list(df.columns)}")
            return None

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            dbg(f"   ⚠️ Column '{col}' has no numeric values in {csv_path}")
            return None

        val = float(values.iloc[-1])

        # Final guard: if metric doesn't sound like time but value is huge -> likely timestamp
        looks_timey_col = _is_timestamp_colname(col)
        looks_timey_metric = metric_name and any(s in metric_name.lower() for s in ["time", "ms", "epoch_time"])
        if not looks_timey_col and not looks_timey_metric and _is_timestamp_value(val):
            dbg(f"   🤨 '{col}' in {csv_path.name} looks like a timestamp ({val}) — ignoring")
            return None

        dbg(f"   ✓ {csv_path.name}: using '{col}' -> last={val}")
        return val

    except Exception as e:
        print(f"⚠️  Warning: Could not read metric CSV {csv_path}: {e}")
        return None

# -------------------- PROV extraction --------------------

def load_prov_json(path: Path) -> ProvDocument:
    with open(path, "r", encoding="utf-8") as f:
        return ProvDocument.deserialize(content=f.read(), format="json")

def extract_metrics_from_entities(prov_doc: ProvDocument, prov_json_path: Path) -> dict:
    """
    Extract metric values by reading the CSV files referenced in metric entities.
    """
    metrics = {}
    base_dir = prov_json_path.parent

    for rec in prov_doc.get_records():
        if not isinstance(rec, ProvEntity):
            continue

        attrs = dict(rec.attributes)
        # Check if metric entity
        is_metric = False
        for k, v in attrs.items():
            if qn_to_prefixed(k) == "yProv4ML:source" and "METRIC" in str(v):
                is_metric = True
                break
        if not is_metric:
            continue

        # Name + path
        metric_name = None
        metric_path = None
        for k, v in attrs.items():
            k_str = qn_to_prefixed(k)
            if k_str == "yProv4ML:label":
                metric_name = str(v)
            elif k_str == "yProv4ML:path":
                metric_path = str(v)

        if not metric_name or not metric_path:
            continue

        # Make absolute/resolved path
        csv_path = Path(metric_path)
        if not csv_path.is_absolute():
            candidate = base_dir / metric_path
            csv_path = candidate if candidate.exists() else Path(metric_path)

        value = read_metric_csv(csv_path, metric_name=metric_name)
        if value is not None:
            metrics[metric_name] = value

    return metrics

def prov_activity_to_row(activity: ProvActivity, prov_doc: ProvDocument, prov_json_path: Path):
    """Extract attributes + artifacts + metric values from CSVs."""
    attrs = dict(activity.attributes)
    row = {}

    for k, v in attrs.items():
        k_str = qn_to_prefixed(k)
        row[normalize_key(k_str)] = v

    row["exp"] = row.get("context_experiment_name") or row.get("experiment_name")
    row["run_id"] = row.get("context_run_id") or row.get("run_id")

    # artifacts
    arts = []
    for rec in prov_doc.get_records():
        if isinstance(rec, ProvEntity):
            ent_attrs = dict(rec.attributes)
            for k2, v2 in ent_attrs.items():
                if qn_to_prefixed(k2) == "context:artifact_path":
                    arts.append(str(v2))
                    break
    if arts:
        row["artifacts"] = ";".join(sorted(set(arts)))

    # metrics via CSVs
    metrics = extract_metrics_from_entities(prov_doc, prov_json_path)
    row.update(metrics)

    return row

def iter_all_prov_jsons(base_root: Path):
    base_root = base_root.resolve()
    if base_root.name == "prov" and base_root.parent.name == "data":
        patterns = [str(base_root / "*" / "*" / "prov_*.json")]
    else:
        patterns = [str(base_root / "*" / "prov_*.json")]

    for pat in patterns:
        for p in glob.iglob(pat, recursive=False):
            yield Path(p)

def infer_exp_version(base_root: Path, prov_json: Path):
    rel = prov_json.resolve().relative_to(base_root.resolve())
    parts = rel.parts
    if len(parts) == 2:
        return base_root.name, parts[0]
    if len(parts) == 3:
        return parts[0], parts[1]
    return None, None

def safe_filename(name: str) -> str:
    if name is None or str(name).strip() == "":
        return "unknown_experiment"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")

def build_dataframe(base_root: Path) -> pd.DataFrame | None:
    rows = []
    for prov_json in iter_all_prov_jsons(base_root):
        exp_name, exp_version = infer_exp_version(base_root, prov_json)

        d = load_prov_json(prov_json)
        activities = [r for r in d.get_records() if isinstance(r, ProvActivity)]
        if not activities:
            continue

        a = activities[0]
        row = prov_activity_to_row(a, d, prov_json)

        if not row.get("exp") and exp_name:
            row["exp"] = exp_name

        row["exp_folder"] = exp_name
        row["exp_version"] = exp_version
        row["source_file"] = str(prov_json)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # keep a minimal expected skeleton; metrics will be added dynamically
    expected = [
        "exp","ACC_val","MSE_train","MSE_val","cpu_energy","cpu_power","cpu_usage",
        "disk_usage","emissions","emissions_rate","energy_consumed","gpu_energy","gpu_power",
        "memory_usage","param_batch_size","param_epochs","param_lr","param_seed","ram_energy",
        "ram_power","train_epoch_time_ms","epochs","lr_min","lr_max","artifacts","run_id",
        "exp_folder","exp_version",    "emissions", "emissions_rate", "energy_consumed",
    "cpu_energy", "gpu_energy", "ram_energy",
    "cpu_power", "gpu_power", "ram_power",
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    other_cols = [c for c in df.columns if c not in expected]
    df = df[expected + other_cols]
    return df

def write_split_by_exp(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    key = "exp" if "exp" in df.columns else "exp_folder"
    file_count = 0

    for exp_value, g in df.groupby(key, dropna=False, sort=True):
        stem = safe_filename(exp_value if pd.notna(exp_value) else "unknown")
        out_path = out_dir / f"{stem}.csv"
        g.to_csv(out_path, index=False)
        file_count += 1
        print(f"✅ Wrote {out_path} with {len(g)} runs and {len(g.columns)} columns")

        # Show which common metrics were populated
        metric_cols = [c for c in g.columns if c in [
            "ACC_val", "MSE_train", "MSE_val", "train_epoch_time_ms",
            "train_loss", "val_loss", "val_accuracy", "learning_rate",
            "cpu_usage", "memory_usage", "disk_usage", "lr_min", "lr_max"
        ]]
        non_empty = [c for c in metric_cols if g[c].notna().any()]
        if non_empty:
            print(f"   Metrics with values: {', '.join(non_empty)}")
        else:
            print(f"   ⚠️  No metric values found (check CSV files exist)")

    print(f"\n✅ Created {file_count} per-experiment CSV file(s) in {out_dir}")

def main(base_root: Path, out_dir: Path):
    print(f"Processing PROV files from: {base_root}")
    print(f"Output directory: {out_dir}")
    print()

    df = build_dataframe(base_root)
    if df is None:
        print(f"❌ No runs/activities found in PROV JSON files under {base_root}.")
        return

    print(f"Found {len(df)} runs total")
    write_split_by_exp(df, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate PROV JSONs into per-experiment CSVs.")
    parser.add_argument("--root", type=Path, default=BASE_ROOT,
                        help="Base root containing experiment folders (default: data/prov)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Directory to write per-experiment CSVs (default: data/unified)")
    parser.add_argument("--debug", action="store_true", help="Verbose metric-column selection logs")
    args = parser.parse_args()
    DEBUG = args.debug
    main(args.root, args.out_dir)
