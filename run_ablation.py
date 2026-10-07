#!/usr/bin/env python3
"""
Run ablation experiments (Tables 1-2) and sensitivity experiments (Tables 5-7)
for the JOC paper revision.
"""
import subprocess, re, os, sys, tempfile, json
from datetime import datetime

PYTHON = "/data/miniconda/envs/torch/bin/python"
CWD = "/data/coding/FECA_final"
BASE_CONFIG = "config/model_config.yaml"
SEED = 2026

DATASETS = ["hdf-9580", "hdf-11000", "healthcaremagic-mimiciii-12k", "healthcaremagic-mimiciii"]

# FECA-optimized base config (Full model)
FULL_CONFIG = {
    "seed": SEED, "embedding_size": 128, "n_layers": 2, "reg_weight": 3.0e-06,
    "train_batch_size": 1024, "learning_rate": 0.001, "knn_k": 17, "temperature": 0.3,
    "alignment_loss_weight": 0.1, "cl_loss_weight": 0.001, "enable_edge_embedding": 0,
    "exposure_penalty_weight": 0.05,
    "enable_pu_score_adjustment": True, "pu_score_weight": 0.04, "pu_penalty_mode": "time_only",
}

METRICS = ["recall@10","mrr@10","ndcg@10","hit@10","precision@10",
           "patientutility@10","doctorutility@10","averageprice@10","averagetime@10",
           "itemcoverage@10","averagepopularity@10","giniindex@10","tailpercentage@10"]

def run_feca(dataset, config_override):
    cfg = {**FULL_CONFIG, **config_override}
    fd, cfgfile = tempfile.mkstemp(suffix=".yaml", dir=CWD)
    with os.fdopen(fd, "w") as f:
        for k, v in cfg.items():
            if isinstance(v, bool): f.write(f"{k}: {'true' if v else 'false'}\n")
            elif isinstance(v, str): f.write(f"{k}: '{v}'\n")
            else: f.write(f"{k}: {v}\n")
    cmd = [PYTHON, "run_customize_model.py", "--model", "FECA",
           "--dataset", dataset, "--config", BASE_CONFIG, "--config", cfgfile]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=CWD)
        for line in (r.stdout + r.stderr).split("\n"):
            if "test result" in line.lower():
                m = {x.group(1).lower(): float(x.group(2))
                     for x in re.finditer(r"'(\w+@\d+)':\s*([0-9.]+)", line)}
                if m: return m
        return None
    except: return None
    finally: os.unlink(cfgfile)

# ============ TABLE 1: Module-level ablation ============
TABLE1_CONFIGS = {
    "Full": {},
    "w.o. UDA": {"alignment_loss_weight": 0.0, "exposure_penalty_weight": 0.0},
    "w.o. PTC": {"enable_pu_score_adjustment": False, "pu_score_weight": 0.0},
    "w.o. UDA & PTC": {"alignment_loss_weight": 0.0, "exposure_penalty_weight": 0.0,
                        "enable_pu_score_adjustment": False, "pu_score_weight": 0.0},
}

# ============ TABLE 2: Loss function ablation ============
TABLE2_CONFIGS = {
    "Full loss": {},
    "w.o. LC": {"cl_loss_weight": 0.0},
    "w.o. LA": {"alignment_loss_weight": 0.0},
    "w.o. LE": {"exposure_penalty_weight": 0.0},
    "w.o. LA + LE": {"alignment_loss_weight": 0.0, "exposure_penalty_weight": 0.0},
    "w.o. Lreg": {"reg_weight": 0.0},
}

# ============ TABLE 5: KNN k sensitivity ============
TABLE5_K_VALUES = [5, 10, 15, 17, 20, 25, 30, 40, 50]

# ============ TABLE 6: GCN layers n sensitivity ============
TABLE6_N_VALUES = [1, 2, 3, 4, 5]

# ============ TABLE 7: Pu/Tu personalization ============
TABLE7_CONFIGS = {
    "Fixed": {"pu_score_weight": 0.04, "pu_penalty_mode": "time_only"},
    "Uniform": {"pu_score_weight": 0.04, "pu_penalty_mode": "time_only", "pu_distribution": "uniform"},
    "Normal": {"pu_score_weight": 0.04, "pu_penalty_mode": "time_only", "pu_distribution": "normal"},
}

def run_table(table_name, configs, datasets=None):
    if datasets is None:
        datasets = DATASETS
    results = {}
    print(f"\n{'='*70}\n  {table_name}\n{'='*70}")
    for ds in datasets:
        results[ds] = {}
        for name, override in configs.items():
            print(f"  {ds} / {name}...", end=" ", flush=True)
            m = run_feca(ds, override)
            if m:
                parts = [f"{k}={v:.4f}" for k, v in m.items()]
                print(" | ".join(parts))
                results[ds][name] = m
            else:
                print("FAILED")
                results[ds][name] = None
    return results

if __name__ == "__main__":
    all_results = {}

    # Determine which tables to run
    tables_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "5", "6"]

    if "1" in tables_to_run:
        all_results["table1"] = run_table("Table 1: Module Ablation", TABLE1_CONFIGS)

    if "2" in tables_to_run:
        all_results["table2"] = run_table("Table 2: Loss Ablation", TABLE2_CONFIGS)

    if "5" in tables_to_run:
        t5_configs = {f"k={k}": {"knn_k": k} for k in TABLE5_K_VALUES}
        all_results["table5"] = run_table("Table 5: KNN k Sensitivity", t5_configs)

    if "6" in tables_to_run:
        t6_configs = {f"n={n}": {"n_layers": n} for n in TABLE6_N_VALUES}
        all_results["table6"] = run_table("Table 6: GCN Layers Sensitivity", t6_configs)

    if "7" in tables_to_run:
        all_results["table7"] = run_table("Table 7: Pu/Tu Personalization", TABLE7_CONFIGS)

    # Save results
    outfile = os.path.join(CWD, "feca_results", "ablation_results.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {outfile}")
