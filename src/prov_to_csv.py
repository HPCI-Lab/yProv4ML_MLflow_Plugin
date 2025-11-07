# src/prov_to_csv.py
import argparse
import glob
from pathlib import Path
import pandas as pd
from prov.model import ProvDocument, QualifiedName, ProvActivity, ProvEntity

BASE_ROOT = Path("data/prov")                 # dynamic root
OUT_CSV = Path("data/unified/all_runs.csv")

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

def prov_activity_to_row(activity: ProvActivity, prov_doc: ProvDocument):
    attrs = dict(activity.attributes)
    row = {}
    for k, v in attrs.items():
        k_str = qn_to_prefixed(k)
        row[normalize_key(k_str)] = v

    row["exp"] = row.get("context_experiment_name") or row.get("experiment_name")
    row["run_id"] = row.get("context_run_id") or row.get("run_id")

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
    return row

# ---------- ADD THESE ----------
def iter_all_prov_jsons(base_root: Path):
    """
    Yield PROV files that match:
      data/prov/<exp>/<version>/prov_*.json
    Works when base_root is either data/prov or data/prov/<exp>.
    """
    base_root = base_root.resolve()
    # If root is .../data/prov -> */*/prov_*.json
    if base_root.name == "prov" and base_root.parent.name == "data":
        patterns = [str(base_root / "*" / "*" / "prov_*.json")]
    else:
        # If root is .../data/prov/<exp> -> */prov_*.json
        patterns = [str(base_root / "*" / "prov_*.json")]

    for pat in patterns:
        for p in glob.iglob(pat, recursive=False):
            yield Path(p)

def infer_exp_version(base_root: Path, prov_json: Path):
    """Return (exp_name, exp_version) regardless of root depth."""
    rel = prov_json.resolve().relative_to(base_root.resolve())
    parts = rel.parts
    if len(parts) == 2:
        # root=data/prov/<exp> => parts=[<version>, prov_*.json]
        return base_root.name, parts[0]
    if len(parts) == 3:
        # root=data/prov => parts=[<exp>, <version>, prov_*.json]
        return parts[0], parts[1]
    return None, None
# ---------- /ADD THESE ----------

def main(base_root: Path, out_csv: Path):
    rows = []
    for prov_json in iter_all_prov_jsons(base_root):
        exp_name, exp_version = infer_exp_version(base_root, prov_json)

        d = load_prov_json(prov_json)
        activities = [r for r in d.get_records() if isinstance(r, ProvActivity)]
        if not activities:
            continue

        a = activities[0]
        row = prov_activity_to_row(a, d)

        if not row.get("exp") and exp_name:
            row["exp"] = exp_name
        row["exp_folder"] = exp_name
        row["exp_version"] = exp_version
        row["source_file"] = str(prov_json)
        rows.append(row)

    if not rows:
        print(f"No runs/activities found in PROV JSON files under {base_root}.")
        return

    df = pd.DataFrame(rows)

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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} runs and {len(df.columns)} columns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate PROV JSONs into a single CSV.")
    parser.add_argument("--root", type=Path, default=BASE_ROOT,
                        help="Base root containing experiment folders (default: data/prov)")
    parser.add_argument("--out", type=Path, default=OUT_CSV,
                        help="Output CSV path (default: data/unified/all_runs.csv)")
    args = parser.parse_args()
    main(args.root, args.out)
