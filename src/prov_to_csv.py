# src/prov_to_csv.py
import glob
from pathlib import Path
import pandas as pd
from prov.model import ProvDocument, QualifiedName, ProvActivity, ProvEntity  # ← import classes

IN_ROOT = Path("data/prov")
OUT_CSV = Path("data/unified/all_runs.csv")

def load_prov_json(path: Path) -> ProvDocument:
    # ProvDocument.deserialize expects a string/bytes and a format
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

    # friendly columns
    row["exp"] = row.get("context_experiment_name") or row.get("experiment_name")
    row["run_id"] = row.get("context_run_id") or row.get("run_id")

    # collect artifact paths
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

def main():
    rows = []
    for prov_json in glob.glob(str(IN_ROOT / "**" / "*.json"), recursive=True):
        d = load_prov_json(Path(prov_json))
        # Pick the first activity in the doc (the run activity your exporter writes)
        activities = [r for r in d.get_records() if isinstance(r, ProvActivity)]
        if not activities:
            continue
        a = activities[0]
        row = prov_activity_to_row(a, d)
        row["source_file"] = prov_json
        rows.append(row)

    if not rows:
        print("No runs/activities found in PROV JSON files.")
        return

    df = pd.DataFrame(rows)

    # Ensure expected columns exist (even if missing in some runs)
    expected = [
        "exp","ACC_val","MSE_train","MSE_val","cpu_energy","cpu_power","cpu_usage",
        "disk_usage","emissions","emissions_rate","energy_consumed","gpu_energy","gpu_power",
        "memory_usage","param_batch_size","param_epochs","param_lr","param_seed","ram_energy",
        "ram_power","train_epoch_time_ms","epochs","lr_min","lr_max","artifacts","run_id"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    other_cols = [c for c in df.columns if c not in expected]
    df = df[expected + other_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df)} runs and {len(df.columns)} columns")

if __name__ == "__main__":
    main()
