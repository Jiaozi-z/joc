#!/usr/bin/env python3
"""
Regenerate all 4 reports with DoctorUtility metric and new baselines (MetaFairBPR, CofairDSPT1).
"""
import subprocess, re, os, tempfile
from datetime import datetime
from collections import OrderedDict

PYTHON = "/data/miniconda/envs/torch/bin/python"
CWD = "/data/coding/FECA/FECA_simple"
BASE_CONFIG = "config/model_config.yaml"
SEED = 2026

DATASETS = ["hdf-9580", "hdf-11000", "healthcaremagic-mimiciii-12k", "healthcaremagic-mimiciii"]

FECA_CONFIGS = {
    "hdf-9580": {"pu_score_weight": 0.04, "pu_penalty_mode": "time_only"},
    "hdf-11000": {"pu_score_weight": 0.30, "pu_penalty_mode": "both"},
    "healthcaremagic-mimiciii-12k": {"pu_score_weight": 0.04, "pu_penalty_mode": "time_only"},
    "healthcaremagic-mimiciii": {"pu_score_weight": 0.04, "pu_penalty_mode": "time_only"},
}

FECA_BASE = {
    "seed": SEED, "embedding_size": 128, "n_layers": 2, "reg_weight": 3.0e-06,
    "train_batch_size": 1024, "learning_rate": 0.001, "knn_k": 17, "temperature": 0.3,
    "alignment_loss_weight": 0.015, "cl_loss_weight": 0.0, "enable_edge_embedding": 0,
    "exposure_penalty_weight": 0.0, "enable_pu_score_adjustment": True,
}

BASELINES = ["RecVAE", "MacridVAE", "DiffRec", "MultiVAE", "NAIS", "RaCT", "GCMC", "LightGCN", "SimGCL", "MetaFairBPR", "CofairDSPT1"]

METRIC_DISPLAY = OrderedDict([
    ("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "NDCG@10"),
    ("hit@10", "Hit@10"), ("precision@10", "Precision@10"),
    ("patientutility@10", "PatientUtility@10"), ("doctorutility@10", "DoctorUtility@10"),
    ("averageprice@10", "AveragePrice@10"), ("averagetime@10", "AverageTime@10"),
    ("itemcoverage@10", "ItemCoverage@10"), ("averagepopularity@10", "AvgPopularity@10"),
    ("giniindex@10", "GiniIndex@10"), ("tailpercentage@10", "TailPct@10"),
])

def run_model(model, dataset, params=None, extra_configs=None):
    if params is None:
        params = {"seed": SEED}
    fd, cfg = tempfile.mkstemp(suffix=".yaml", dir=CWD)
    with os.fdopen(fd, "w") as f:
        for k, v in params.items():
            if isinstance(v, bool): f.write(f"{k}: {'true' if v else 'false'}\n")
            elif isinstance(v, str): f.write(f"{k}: '{v}'\n")
            elif isinstance(v, list): f.write(f"{k}: {v}\n")
            else: f.write(f"{k}: {v}\n")
    cmd = [PYTHON, "run_customize_model.py", "--model", model, "--dataset", dataset]
    if extra_configs:
        for ec in extra_configs:
            cmd += ["--config", ec]
    cmd += ["--config", BASE_CONFIG, "--config", cfg]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=CWD)
        for line in (r.stdout + r.stderr).split("\n"):
            if "test result" in line.lower():
                m = {x.group(1).lower(): float(x.group(2))
                     for x in re.finditer(r"'(\w+@\d+)':\s*([0-9.]+)", line)}
                if m: return m
        return None
    except: return None
    finally: os.unlink(cfg)

def run_baseline(model, dataset):
    if model == "MetaFairBPR":
        return run_model(model, dataset, {"seed": SEED}, ["config/metafairbpr.yaml"])
    elif model == "CofairDSPT1":
        return run_model(model, dataset, {"seed": SEED}, ["config/cofair_dspt1.yaml"])
    else:
        return run_model(model, dataset, {"seed": SEED})

def write_report(results, dataset, path):
    d = OrderedDict((k, v) for k, v in results.items() if v)
    pu_cfg = FECA_CONFIGS[dataset]
    with open(path, "w") as f:
        f.write(f"# PatientUtility Evaluation on {dataset}\n\n")
        f.write(f"- **Dataset**: {dataset}\n- **Seed**: {SEED}\n")
        f.write(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("- **PatientUtility**: `NDCG × (1 - NormCost) × (1 - NormWaitTime)` (↑ higher = better)\n")
        f.write(f"- **FECA-optimized**: `pu_score_weight={pu_cfg['pu_score_weight']}, pu_penalty_mode={pu_cfg['pu_penalty_mode']}`\n")
        f.write("- **FECA-low-time**: `pu_score_weight=0.30, pu_penalty_mode=time_only`\n")
        f.write("- All baseline models use optimized configs from `recbole/properties/model/`\n\n")
        f.write("## 1. Full Results\n\n")
        cols = list(METRIC_DISPLAY.values())
        f.write("| Model | " + " | ".join(cols) + " |\n")
        f.write("|-------" + "|------:" * len(cols) + "|\n")
        for name, m in d.items():
            vals = [f"{m.get(mk, 0):.4f}" for mk in METRIC_DISPLAY]
            f.write(f"| {name} | " + " | ".join(vals) + " |\n")
        f.write("\n## 2. Rankings\n")
        # PU
        f.write("\n### PatientUtility@10 (↑)\n\n")
        f.write("| Rank | Model | PU@10 | NDCG@10 | AvgPrice@10 | AvgTime@10 |\n")
        f.write("|------|-------|------:|--------:|------------:|-----------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("patientutility@10", 0), reverse=True), 1):
            f.write(f"| {i} | **{n}** | {m.get('patientutility@10',0):.4f} | {m.get('ndcg@10',0):.4f} | {m.get('averageprice@10',0):.2f} | {m.get('averagetime@10',0):.2f} |\n")
        # DU
        f.write("\n### DoctorUtility@10 (↑)\n\n")
        f.write("| Rank | Model | DU@10 |\n|------|-------|------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("doctorutility@10", 0), reverse=True), 1):
            f.write(f"| {i} | {n} | {m.get('doctorutility@10',0):.4f} |\n")
        # Recall
        f.write("\n### Recall@10 (↑)\n\n| Rank | Model | Value |\n|------|-------|------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("recall@10", 0), reverse=True), 1):
            f.write(f"| {i} | {n} | {m.get('recall@10',0):.4f} |\n")
        # NDCG
        f.write("\n### NDCG@10 (↑)\n\n| Rank | Model | Value |\n|------|-------|------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("ndcg@10", 0), reverse=True), 1):
            f.write(f"| {i} | {n} | {m.get('ndcg@10',0):.4f} |\n")
        # TailPct
        f.write("\n### TailPercentage@10 (↑)\n\n| Rank | Model | Value |\n|------|-------|------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("tailpercentage@10", 0), reverse=True), 1):
            f.write(f"| {i} | {n} | {m.get('tailpercentage@10',0):.4f} |\n")
        # Price
        f.write("\n### AveragePrice@10 (↓)\n\n| Rank | Model | Value |\n|------|-------|------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("averageprice@10", 999)), 1):
            f.write(f"| {i} | {n} | {m.get('averageprice@10',0):.4f} |\n")
        # Time
        f.write("\n### AverageTime@10 (↓)\n\n| Rank | Model | Value |\n|------|-------|------:|\n")
        for i, (n, m) in enumerate(sorted(d.items(), key=lambda x: x[1].get("averagetime@10", 999)), 1):
            f.write(f"| {i} | {n} | {m.get('averagetime@10',0):.4f} |\n")
    print(f"  Report: {path}")

import sys
datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else DATASETS

for dataset in datasets_to_run:
    print(f"\n{'='*60}\n  Dataset: {dataset}\n{'='*60}")
    results = OrderedDict()
    # FECA variants
    print("  Running FECA...", flush=True)
    results["FECA"] = run_model("FECA", dataset, {"seed": SEED})
    pu_cfg = FECA_CONFIGS[dataset]
    print("  Running FECA-optimized...", flush=True)
    results["FECA-optimized"] = run_model("FECA", dataset, {**FECA_BASE, **pu_cfg})
    print("  Running FECA-low-time...", flush=True)
    results["FECA-low-time"] = run_model("FECA", dataset, {**FECA_BASE, "pu_score_weight": 0.30, "pu_penalty_mode": "time_only"})
    # Baselines
    for model in BASELINES:
        print(f"  Running {model}...", flush=True)
        results[model] = run_baseline(model, dataset)
    # Summary
    valid = {k: v for k, v in results.items() if v}
    if valid:
        pu1 = max(valid.items(), key=lambda x: x[1].get("patientutility@10", 0))
        r1 = max(valid.items(), key=lambda x: x[1].get("recall@10", 0))
        du1 = max(valid.items(), key=lambda x: x[1].get("doctorutility@10", 0))
        print(f"  PU#1: {pu1[0]} ({pu1[1].get('patientutility@10',0):.4f})")
        print(f"  Recall#1: {r1[0]} ({r1[1].get('recall@10',0):.4f})")
        print(f"  DU#1: {du1[0]} ({du1[1].get('doctorutility@10',0):.4f})")
    out = os.path.join(CWD, "feca_results", f"{dataset}_seed{SEED}_patientutility.md")
    write_report(results, dataset, out)
