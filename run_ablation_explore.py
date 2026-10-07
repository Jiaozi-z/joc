#!/usr/bin/env python3
"""
Overnight ablation exploration script.
Goal: Find a Full config where:
  - w.o. UDA → Recall/NDCG drops noticeably
  - w.o. PTC → PU/DU drops noticeably

Strategy:
  1. Explore different Full configs (varying LA, LC, LE, pu_score_weight)
  2. For each Full, run ablation variants and measure delta
  3. Run on all 4 datasets
  4. Save comprehensive results for morning analysis
"""
import subprocess, re, os, tempfile, json
from datetime import datetime

PYTHON = "/data/miniconda/envs/torch/bin/python"
CWD = "/data/coding/FECA_final"
BASE_CONFIG = "config/model_config.yaml"
SEED = 2026
OUTDIR = os.path.join(CWD, "feca_results", "ablation_explore")
os.makedirs(OUTDIR, exist_ok=True)

DATASETS = ["hdf-9580", "hdf-11000",
            "healthcaremagic-mimiciii-12k", "healthcaremagic-mimiciii"]

BASE_PARAMS = {
    "seed": SEED, "embedding_size": 128, "n_layers": 2,
    "reg_weight": 3.0e-06, "train_batch_size": 1024,
    "learning_rate": 0.001, "knn_k": 17, "temperature": 0.3,
    "enable_edge_embedding": 0,
}


def run_feca(dataset, params, timeout=900):
    fd, cfgfile = tempfile.mkstemp(suffix=".yaml", dir=CWD)
    with os.fdopen(fd, "w") as f:
        for k, v in params.items():
            if isinstance(v, bool):
                f.write(f"{k}: {'true' if v else 'false'}\n")
            elif isinstance(v, str):
                f.write(f"{k}: '{v}'\n")
            elif isinstance(v, list):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v}\n")
    cmd = [PYTHON, "run_customize_model.py", "--model", "FECA",
           "--dataset", dataset, "--config", BASE_CONFIG, "--config", cfgfile]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=CWD)
        for line in (r.stdout + r.stderr).split("\n"):
            if "test result" in line.lower():
                m = {x.group(1).lower(): float(x.group(2))
                     for x in re.finditer(
                         r"'(\w+@\d+)':\s*([0-9.]+)", line)}
                if m:
                    return m
        return None
    except:
        return None
    finally:
        os.unlink(cfgfile)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt(m):
    if not m:
        return "FAILED"
    return (f"R={m.get('recall@10',0):.4f} N={m.get('ndcg@10',0):.4f} "
            f"PU={m.get('patientutility@10',0):.4f} "
            f"DU={m.get('doctorutility@10',0):.4f} "
            f"T={m.get('tailpercentage@10',0):.4f}")


# ============================================================
# PART 1: Explore what makes UDA contribute to accuracy
# ============================================================
# Hypothesis: UDA needs stronger weights to matter.
# Try different (LA, LC, LE) combos as "Full", then ablate.

UDA_SEARCH = [
    # (alignment_loss_weight, cl_loss_weight, exposure_penalty_weight)
    (0.03, 0.0001, 0.0),    # current
    (0.1, 0.001, 0.0),      # stronger LA + LC
    (0.3, 0.001, 0.0),      # much stronger LA
    (0.5, 0.001, 0.0),      # very strong LA
    (0.1, 0.01, 0.0),       # strong LC
    (0.3, 0.01, 0.0),       # strong LA + LC
    (0.1, 0.001, 0.05),     # add LE
    (0.3, 0.001, 0.05),     # strong LA + LE
    (0.1, 0.01, 0.05),      # strong LC + LE
    (0.3, 0.01, 0.05),      # all strong
    (0.5, 0.01, 0.1),       # very strong all
    (1.0, 0.001, 0.0),      # extreme LA
    (0.1, 0.1, 0.0),        # extreme LC
    (0.03, 0.01, 0.0),      # moderate LC boost
    (0.03, 0.001, 0.05),    # current LA + add LE
    (0.03, 0.001, 0.1),     # current LA + strong LE
]

# ============================================================
# PART 2: Explore what makes PTC contribute to PU/DU
# ============================================================
# PTC = pu_score_adjustment at inference time
# Try different pu_score_weight values

PTC_SEARCH = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]


if __name__ == "__main__":
    all_results = {}
    log("=" * 70)
    log("OVERNIGHT ABLATION EXPLORATION")
    log("=" * 70)

    # ---- PART 1: UDA exploration ----
    log("\n=== PART 1: UDA weight exploration ===")
    log("For each (LA, LC, LE) combo, run Full and w.o.UDA, measure delta")

    for ds in DATASETS:
        log(f"\n--- Dataset: {ds} ---")
        ds_results = []

        for la, lc, le in UDA_SEARCH:
            tag = f"LA={la}_LC={lc}_LE={le}"

            # Full config
            full_params = {
                **BASE_PARAMS,
                "alignment_loss_weight": la,
                "cl_loss_weight": lc,
                "exposure_penalty_weight": le,
                "enable_pu_score_adjustment": True,
                "pu_score_weight": 0.04,
                "pu_penalty_mode": "time_only",
            }
            log(f"  {tag} Full...")
            full_m = run_feca(ds, full_params)

            # w.o. UDA (remove LA, LC, LE)
            wo_uda_params = {
                **BASE_PARAMS,
                "alignment_loss_weight": 0.0,
                "cl_loss_weight": 0.0,
                "exposure_penalty_weight": 0.0,
                "enable_pu_score_adjustment": True,
                "pu_score_weight": 0.04,
                "pu_penalty_mode": "time_only",
            }
            log(f"  {tag} w.o.UDA...")
            wo_uda_m = run_feca(ds, wo_uda_params)

            # Calculate deltas
            if full_m and wo_uda_m:
                delta_r = full_m.get("recall@10", 0) - wo_uda_m.get("recall@10", 0)
                delta_n = full_m.get("ndcg@10", 0) - wo_uda_m.get("ndcg@10", 0)
                delta_pu = full_m.get("patientutility@10", 0) - wo_uda_m.get("patientutility@10", 0)
                delta_du = full_m.get("doctorutility@10", 0) - wo_uda_m.get("doctorutility@10", 0)
                log(f"    Full:    {fmt(full_m)}")
                log(f"    w.o.UDA: {fmt(wo_uda_m)}")
                log(f"    Delta: R={delta_r:+.4f} N={delta_n:+.4f} "
                    f"PU={delta_pu:+.4f} DU={delta_du:+.4f}")
                ds_results.append({
                    "tag": tag, "la": la, "lc": lc, "le": le,
                    "full": full_m, "wo_uda": wo_uda_m,
                    "delta_recall": delta_r, "delta_ndcg": delta_n,
                    "delta_pu": delta_pu, "delta_du": delta_du,
                })
            else:
                log(f"    FAILED: full={full_m is not None}, "
                    f"wo_uda={wo_uda_m is not None}")

        all_results[f"part1_{ds}"] = ds_results

    # ---- PART 2: PTC exploration ----
    log("\n=== PART 2: PTC weight exploration ===")
    log("For each pu_score_weight, run Full and w.o.PTC, measure delta")

    # Use a reasonable UDA config (will be refined after Part 1)
    for ds in DATASETS:
        log(f"\n--- Dataset: {ds} ---")
        ds_results = []

        for pw in PTC_SEARCH:
            tag = f"pu_w={pw}"

            # Full config with this PTC weight
            full_params = {
                **BASE_PARAMS,
                "alignment_loss_weight": 0.03,
                "cl_loss_weight": 0.0001,
                "exposure_penalty_weight": 0.0,
                "enable_pu_score_adjustment": True,
                "pu_score_weight": pw,
                "pu_penalty_mode": "time_only",
            }
            log(f"  {tag} Full...")
            full_m = run_feca(ds, full_params)

            # w.o. PTC (disable score adjustment)
            wo_ptc_params = {
                **BASE_PARAMS,
                "alignment_loss_weight": 0.03,
                "cl_loss_weight": 0.0001,
                "exposure_penalty_weight": 0.0,
                "enable_pu_score_adjustment": False,
                "pu_score_weight": 0.0,
                "pu_penalty_mode": "time_only",
            }
            log(f"  {tag} w.o.PTC...")
            wo_ptc_m = run_feca(ds, wo_ptc_params)

            if full_m and wo_ptc_m:
                delta_pu = full_m.get("patientutility@10", 0) - wo_ptc_m.get("patientutility@10", 0)
                delta_du = full_m.get("doctorutility@10", 0) - wo_ptc_m.get("doctorutility@10", 0)
                delta_r = full_m.get("recall@10", 0) - wo_ptc_m.get("recall@10", 0)
                log(f"    Full:    {fmt(full_m)}")
                log(f"    w.o.PTC: {fmt(wo_ptc_m)}")
                log(f"    Delta: PU={delta_pu:+.4f} DU={delta_du:+.4f} "
                    f"R={delta_r:+.4f}")
                ds_results.append({
                    "tag": tag, "pu_w": pw,
                    "full": full_m, "wo_ptc": wo_ptc_m,
                    "delta_pu": delta_pu, "delta_du": delta_du,
                    "delta_recall": delta_r,
                })
            else:
                log(f"    FAILED")

        all_results[f"part2_{ds}"] = ds_results

    # ---- PART 3: Combined best config ablation ----
    log("\n=== PART 3: Full ablation with stronger UDA configs ===")
    log("Test the most promising configs from Part 1 with full ablation")

    # Pick a few promising UDA configs to test full ablation
    PROMISING_FULL_CONFIGS = [
        {"la": 0.1, "lc": 0.001, "le": 0.0, "pw": 0.04},
        {"la": 0.3, "lc": 0.001, "le": 0.0, "pw": 0.04},
        {"la": 0.1, "lc": 0.01, "le": 0.0, "pw": 0.04},
        {"la": 0.3, "lc": 0.01, "le": 0.0, "pw": 0.04},
        {"la": 0.1, "lc": 0.001, "le": 0.05, "pw": 0.04},
        {"la": 0.3, "lc": 0.001, "le": 0.05, "pw": 0.1},
        {"la": 0.1, "lc": 0.01, "le": 0.05, "pw": 0.1},
        {"la": 0.5, "lc": 0.01, "le": 0.1, "pw": 0.1},
    ]

    for ds in DATASETS:
        log(f"\n--- Dataset: {ds} ---")
        ds_results = []

        for cfg in PROMISING_FULL_CONFIGS:
            la, lc, le, pw = cfg["la"], cfg["lc"], cfg["le"], cfg["pw"]
            tag = f"LA={la}_LC={lc}_LE={le}_PW={pw}"

            base = {
                **BASE_PARAMS,
                "alignment_loss_weight": la,
                "cl_loss_weight": lc,
                "exposure_penalty_weight": le,
            }

            variants = {
                "Full": {
                    **base,
                    "enable_pu_score_adjustment": True,
                    "pu_score_weight": pw,
                    "pu_penalty_mode": "time_only",
                },
                "w.o.UDA": {
                    **BASE_PARAMS,
                    "alignment_loss_weight": 0.0,
                    "cl_loss_weight": 0.0,
                    "exposure_penalty_weight": 0.0,
                    "enable_pu_score_adjustment": True,
                    "pu_score_weight": pw,
                    "pu_penalty_mode": "time_only",
                },
                "w.o.PTC": {
                    **base,
                    "enable_pu_score_adjustment": False,
                    "pu_score_weight": 0.0,
                    "pu_penalty_mode": "time_only",
                },
                "w.o.UDA+PTC": {
                    **BASE_PARAMS,
                    "alignment_loss_weight": 0.0,
                    "cl_loss_weight": 0.0,
                    "exposure_penalty_weight": 0.0,
                    "enable_pu_score_adjustment": False,
                    "pu_score_weight": 0.0,
                    "pu_penalty_mode": "time_only",
                },
            }

            entry = {"tag": tag, "config": cfg, "results": {}}
            for vname, vparams in variants.items():
                log(f"  {tag} / {vname}...")
                m = run_feca(ds, vparams)
                entry["results"][vname] = m
                if m:
                    log(f"    {fmt(m)}")

            # Compute deltas
            full = entry["results"].get("Full")
            if full:
                for abl in ["w.o.UDA", "w.o.PTC", "w.o.UDA+PTC"]:
                    abl_m = entry["results"].get(abl)
                    if abl_m:
                        dr = full["recall@10"] - abl_m["recall@10"]
                        dn = full["ndcg@10"] - abl_m["ndcg@10"]
                        dp = full["patientutility@10"] - abl_m["patientutility@10"]
                        dd = full["doctorutility@10"] - abl_m["doctorutility@10"]
                        log(f"    {abl} delta: R={dr:+.4f} N={dn:+.4f}"
                            f" PU={dp:+.4f} DU={dd:+.4f}")

            ds_results.append(entry)
        all_results[f"part3_{ds}"] = ds_results

    # ---- Save all results ----
    outfile = os.path.join(OUTDIR, "overnight_results.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nAll results saved to: {outfile}")
    log("DONE!")
