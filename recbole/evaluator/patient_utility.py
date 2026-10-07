# -*- encoding: utf-8 -*-
r"""
Patient Utility Metric
########################

Patient Utility incorporates clinical outcomes, economic costs, and waiting time burdens
following the framework of Brown et al. (2000) and Winzenborg et al. (2020).
"""

import numpy as np
import torch
from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType


class PatientUtility(AbstractMetric):
    r"""PatientUtility computes the patient utility incorporating clinical outcomes,
    economic costs, and waiting time burdens.

    Following Brown et al. (2000) and Winzenborg et al. (2020), patient utility is:

    .. math::
        U_p(K) = MatchQuality(p,K) \times (1 - Cost(K)) \times (1 - WaitTime(K))

    Where:
    - MatchQuality(p,K) = NDCG@K (clinical compatibility)
    - Cost(K) = normalized AveragePrice@K
    - WaitTime(K) = normalized AverageTime@K

    Higher values indicate better patient utility.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "rec.topk", "data.num_items", "rec.score",
                   "data.item_price", "data.item_time"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get all necessary information for patient utility calculation"""
        # For NDCG calculation (MatchQuality) - use TopkMetric's method
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        pos_idx = topk_idx.to(torch.bool).numpy()
        pos_len = pos_len_list.squeeze(-1).numpy()

        # For price and time
        item_price = dataobject.get("data.item_price")
        item_time = dataobject.get("data.item_time")
        item_matrix = dataobject.get("rec.items").numpy()

        return pos_idx, pos_len, item_matrix, item_price, item_time

    def get_pos_index(self, rec_mat, pos_len_list):
        """Get position index for NDCG calculation - DEPRECATED, use used_info instead"""
        pos_idx = np.zeros_like(rec_mat.numpy(), dtype=np.int32)
        for i, length in enumerate(pos_len_list):
            length_int = int(length.item()) if hasattr(length, 'item') else int(length)
            pos_idx[i][:length_int] = 1
        return pos_idx

    def calculate_metric(self, dataobject):
        pos_idx, pos_len, item_matrix, item_price, item_time = self.used_info(dataobject)

        # Calculate MatchQuality using NDCG
        match_quality = self.calculate_ndcg(pos_idx, pos_len)

        # Calculate normalized Cost
        prices = self.get_prices(item_matrix, item_price)
        avg_prices = prices.cumsum(axis=1) / np.arange(1, prices.shape[1] + 1)
        normalized_cost = self.normalize_values(avg_prices)

        # Calculate normalized WaitTime
        times = self.get_times(item_matrix, item_time)
        avg_times = times.cumsum(axis=1) / np.arange(1, times.shape[1] + 1)
        normalized_time = self.normalize_values(avg_times)

        # Calculate Patient Utility
        # U_p(K) = MatchQuality × (1 - Cost) × (1 - WaitTime)
        utility = match_quality * (1 - normalized_cost) * (1 - normalized_time)

        metric_dict = self.topk_result("patientutility", utility)
        return metric_dict

    def calculate_ndcg(self, pos_index, pos_len):
        """Calculate NDCG for MatchQuality component"""
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float64)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float64)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result

    def get_prices(self, item_matrix, item_price):
        """Convert item matrix to price matrix"""
        value = np.zeros_like(item_matrix, dtype=float)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_price[row[j]].item()
        return value

    def get_times(self, item_matrix, item_time):
        """Convert item matrix to time matrix"""
        value = np.zeros_like(item_matrix, dtype=float)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_time[row[j]].item()
        return value

    def normalize_values(self, values):
        """Normalize values to [0, 1] range using min-max normalization"""
        # Handle each K separately to get proper normalization per topk
        normalized = np.zeros_like(values)
        for k_idx in range(values.shape[1]):
            col = values[:, k_idx]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                normalized[:, k_idx] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, k_idx] = 0.0
        return normalized

    def topk_result(self, metric, value):
        """Format results for different K values"""
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict
