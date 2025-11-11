# src/prov_to_csv.py
import argparse
import glob
import re
from pathlib import Path
import pandas as pd
from prov.model import ProvDocument, QualifiedName, ProvActivity, ProvEntity

BASE_ROOT = Path("data/prov")
OUT_DIR = Path("data/unified")

def load_prov_json(path: Path) -> ProvDocument:
    with open(path, "r", encoding="utf-8") as f:
        return ProvDocument.deserialize(content=f.read(), format="json")

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

def read_metric_csv(csv_path: Path):
    """
    Read a metric CSV file and return aggregated values.
    
    prov4ml metric CSVs typically have columns like:
    - step, value (for step-based metrics)
    - or just value (for single values)
    
    Returns the last value, or mean if multiple values exist.
    """
    try:
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        
        # Try to find value column (various names)
        value_col = None
        for col in ['value', 'Value', 'val', 'metric_value']:
            if col in df.columns:
                value_col = col
                break
        
        if value_col is None and len(df.columns) > 0:
            # Use last column as value
            value_col = df.columns[-1]
        
        if value_col and value_col in df.columns:
            values = df[value_col].dropna()
            if len(values) > 0:
                # Return last value (most recent)
                return float(values.iloc[-1])
        
        return None
    except Exception as e:
        print(f"⚠️  Warning: Could not read metric CSV {csv_path}: {e}")
        return None

def extract_metrics_from_entities(prov_doc: ProvDocument, prov_json_path: Path) -> dict:
    """
    Extract metric values by reading the CSV files referenced in entities.
    
    The PROV JSON has entities like:
    {
        "train_epoch_time_ms": {
            "yProv4ML:path": "data/prov/.../metrics/train_epoch_time_ms_..._GR0.csv",
            "yProv4ML:source": "LoggingItemKind.METRIC"
        }
    }
    
    We need to read those CSV files to get the actual values.
    """
    metrics = {}
    base_dir = prov_json_path.parent  # Directory containing the prov JSON
    
    for rec in prov_doc.get_records():
        if not isinstance(rec, ProvEntity):
            continue
        
        attrs = dict(rec.attributes)
        
        # Check if this is a metric entity
        source = None
        for k, v in attrs.items():
            k_str = qn_to_prefixed(k)
            if k_str == "yProv4ML:source" and "METRIC" in str(v):
                source = "metric"
                break
        
        if source != "metric":
            continue
        
        # Get the metric name (label)
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
        
        # Convert path to absolute
        csv_path = Path(metric_path)
        if not csv_path.is_absolute():
            # Try relative to base_dir
            csv_path = base_dir / metric_path
            if not csv_path.exists():
                # Try relative to project root
                csv_path = Path(metric_path)
        
        # Read the CSV and get the value
        value = read_metric_csv(csv_path)
        if value is not None:
            metrics[metric_name] = value
    
    return metrics

def prov_activity_to_row(activity: ProvActivity, prov_doc: ProvDocument, prov_json_path: Path):
    """Extract data from activity and add metric values from CSV files."""
    attrs = dict(activity.attributes)
    row = {}
    
    # Extract attributes from activity
    for k, v in attrs.items():
        k_str = qn_to_prefixed(k)
        row[normalize_key(k_str)] = v

    row["exp"] = row.get("context_experiment_name") or row.get("experiment_name")
    row["run_id"] = row.get("context_run_id") or row.get("run_id")

    # Extract artifacts
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
    
    # ✅ NEW: Extract metric values from CSV files
    metrics = extract_metrics_from_entities(prov_doc, prov_json_path)
    row.update(metrics)
    
    return row

def iter_all_prov_jsons(base_root: Path):
    """Yield PROV JSON files."""
    base_root = base_root.resolve()
    if base_root.name == "prov" and base_root.parent.name == "data":
        patterns = [str(base_root / "*" / "*" / "prov_*.json")]
    else:
        patterns = [str(base_root / "*" / "prov_*.json")]

    for pat in patterns:
        for p in glob.iglob(pat, recursive=False):
            yield Path(p)

def infer_exp_version(base_root: Path, prov_json: Path):
    """Return (exp_name, exp_version)."""
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
        row = prov_activity_to_row(a, d, prov_json)  # ✅ Pass prov_json path

        if not row.get("exp") and exp_name:
            row["exp"] = exp_name

        row["exp_folder"] = exp_name
        row["exp_version"] = exp_version
        row["source_file"] = str(prov_json)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Expected columns (add common metrics)
    expected = [
        "exp","ACC_val","MSE_train","MSE_val","cpu_energy","cpu_power","cpu_usage",
        "disk_usage","emissions","emissions_rate","energy_consumed","gpu_energy","gpu_power",
        "memory_usage","param_batch_size","param_epochs","param_lr","param_seed","ram_energy",
        "ram_power","train_epoch_time_ms","epochs","lr_min","lr_max","artifacts","run_id",
        "exp_folder","exp_version"
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
        
        # Show which metrics were found
        metric_cols = [c for c in g.columns if c in [
            "ACC_val", "MSE_train", "MSE_val", "train_epoch_time_ms", 
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
    args = parser.parse_args()
    main(args.root, args.out_dir)