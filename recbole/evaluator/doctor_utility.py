# -*- encoding: utf-8 -*-
r"""
Doctor Utility Metric
########################

Doctor Utility measures the professional and economic benefits a physician derives
from the recommendation system, incorporating fair exposure and clinical compatibility.
"""

import numpy as np
import torch
from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType


class DoctorUtility(AbstractMetric):
    r"""DoctorUtility computes the doctor utility incorporating fair exposure
    and match quality penalty.

    Doctor utility is defined as:

    .. math::
        U_d(d) = \frac{FairExposure(d)}{\sum_p (1 - MatchQuality(K, d))}

    Where:
    - FairExposure(d) = recommendation visibility for underserved doctors
    - MatchQuality(p,d) = normalized clinical compatibility between patient p and doctor d

    Higher values indicate better doctor utility (fair exposure with good matching).
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "rec.topk", "data.num_items", "rec.score"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get all necessary information for doctor utility calculation"""
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        pos_idx = topk_idx.to(torch.bool).numpy()
        pos_len = pos_len_list.squeeze(-1).numpy()

        item_matrix = dataobject.get("rec.items").numpy()
        num_items = dataobject.get("data.num_items")

        return pos_idx, pos_len, item_matrix, num_items

    def calculate_metric(self, dataobject):
        pos_idx, pos_len, item_matrix, num_items = self.used_info(dataobject)

        result = {}
        for k in self.topk:
            utility = self.compute_doctor_utility(
                pos_idx, pos_len, item_matrix, num_items, k
            )
            key = "doctorutility@{}".format(k)
            result[key] = round(utility, self.decimal_place)

        return result

    def compute_doctor_utility(self, pos_idx, pos_len, item_matrix, num_items, topk):
        """Compute average doctor utility across all doctors at given K"""
        n_users = item_matrix.shape[0]
        actual_k = min(topk, item_matrix.shape[1])

        topk_items = item_matrix[:, :actual_k]

        # --- FairExposure(d): how fairly each doctor is exposed ---
        exposure_count = np.zeros(num_items, dtype=np.float64)
        for i in range(n_users):
            for j in range(actual_k):
                doctor_id = topk_items[i, j]
                exposure_count[doctor_id] += 1.0

        ideal_exposure = n_users * actual_k / num_items
        fair_exposure = np.zeros(num_items, dtype=np.float64)
        for d in range(num_items):
            if ideal_exposure > 0:
                fair_exposure[d] = min(exposure_count[d] / ideal_exposure, 1.0)

        # --- MatchQuality penalty: sum_p (1 - MatchQuality(K, d)) ---
        # MatchQuality is based on NDCG-style relevance per doctor
        match_quality_per_doctor = np.zeros(num_items, dtype=np.float64)
        doctor_patient_count = np.zeros(num_items, dtype=np.float64)

        pos_topk = pos_idx[:, :actual_k]
        for i in range(n_users):
            for j in range(actual_k):
                doctor_id = topk_items[i, j]
                rank = j + 1
                gain = 1.0 / np.log2(rank + 1)
                if pos_topk[i, j]:
                    match_quality_per_doctor[doctor_id] += gain
                doctor_patient_count[doctor_id] += gain

        # Normalize match quality to [0, 1] per doctor
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_mq = np.where(
                doctor_patient_count > 0,
                match_quality_per_doctor / doctor_patient_count,
                0.0
            )

        # Compute penalty: sum of (1 - MatchQuality) across patients assigned to each doctor
        # Use exposure_count as number of patient assignments
        mismatch_penalty = exposure_count * (1.0 - normalized_mq)

        # --- Doctor Utility: FairExposure(d) / sum_p(1 - MatchQuality(K, d)) ---
        with np.errstate(divide='ignore', invalid='ignore'):
            doctor_utilities = np.where(
                mismatch_penalty > 0,
                fair_exposure / mismatch_penalty,
                fair_exposure  # if no mismatch penalty, utility equals fair exposure
            )

        # Average over doctors that have at least one recommendation
        active_doctors = exposure_count > 0
        if active_doctors.sum() > 0:
            avg_utility = doctor_utilities[active_doctors].mean()
        else:
            avg_utility = 0.0

        return float(avg_utility)
