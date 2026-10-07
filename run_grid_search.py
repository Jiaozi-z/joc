#!/usr/bin/env python3
"""
Table 3-4: λC × λA × λE 3D grid search for JOC paper.
8×8×6 = 384 combos per dataset × 4 datasets = 1536 total runs.
"""
import subprocess, re, os, sys, tempfile, json, time
from datetime import datetime
from itertools import product

PYTHON = "/data/miniconda/envs/torch/bin/python"
CWD = "/data/coding/FECA_final"
BASE_CONFIG = "config/model_config.yaml"
SEED = 2026

DATASETS = ["hdf-9580", "hdf-11000", "healthcaremagic-mimiciii-12k", "healthcaremagic-mimiciii"]

# Fixed params (same as ablation FULL_CONFIG minus the three being searched)
FIXED_PARAMS = {
    "seed": SEED, "embedding_size": 128, "n_layers": 2, "reg_weight": 3.0e-06,
    "train_batch_size": 1024, "learning_rate": 0.001, "knn_k": 17, "temperature": 0.3,
    "enable_edge_embedding": 0,
    "enable_pu_score_adjustment": True, "pu_score_weight": 0.04, "pu_penalty_mode": "time_only",
}

# Grid values based on exploratory analysis
LC_VALUES = [0, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.01]
LA_VALUES = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
LE_VALUES = [0, 0.01, 0.03, 0.05, 0.07, 0.1]

METRICS = ["recall@10","mrr@10","ndcg@10","hit@10","precision@10",
           "patientutility@10","doctorutility@10","averageprice@10","averagetime@10",
           "itemcoverage@10","averagepopularity@10","giniindex@10","tailpercentage@10"]


def run_feca(dataset, config_override):
    cfg = {**FIXED_PARAMS, **config_override}
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
                if m:
                    os.unlink(cfgfile)
                    return m
    except Exception as e:
        print(f"    ERROR: {e}")
    try:
        os.unlink(cfgfile)
    except:
        pass
    return None


def main():
    outdir = os.path.join(CWD, "feca_results")
    results_file = os.path.join(outdir, "grid_search_results.json")
    log_file = os.path.join(outdir, "grid_search_log.txt")

    # Resume from existing results if available
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    combos = list(product(LC_VALUES, LA_VALUES, LE_VALUES))
    total_per_ds = len(combos)
    total = total_per_ds * len(DATASETS)
    print(f"Grid search: {len(LC_VALUES)}×{len(LA_VALUES)}×{len(LE_VALUES)} = {total_per_ds}/dataset, {total} total")
    print(f"Started: {datetime.now()}")

    done = 0
    with open(log_file, "a") as log:
        log.write(f"\n{'='*70}\n  Grid Search Started: {datetime.now()}\n{'='*70}\n")
        for ds in DATASETS:
            if ds not in all_results:
                all_results[ds] = {}
            for lc, la, le in combos:
                key = f"lc={lc}_la={la}_le={le}"
                if key in all_results[ds]:
                    done += 1
                    continue
                override = {
                    "cl_loss_weight": lc,
                    "alignment_loss_weight": la,
                    "exposure_penalty_weight": le,
                }
                m = run_feca(ds, override)
                done += 1
                if m:
                    all_results[ds][key] = m
                    parts = [f"{k}={v:.4f}" for k, v in m.items()]
                    line = f"  {ds} / lc={lc}, la={la}, le={le} ... {' | '.join(parts)}"
                    print(f"[{done}/{total}] {line}")
                    log.write(line + "\n")
                    log.flush()
                else:
                    print(f"[{done}/{total}] {ds} / lc={lc}, la={la}, le={le} ... FAILED")
                    log.write(f"  {ds} / lc={lc}, la={la}, le={le} ... FAILED\n")
                    log.flush()

                # Save checkpoint every 10 runs
                if done % 10 == 0:
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)

        # Final save
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone: {datetime.now()}")
    print(f"Results: {results_file}")


if __name__ == "__main__":
    main()
