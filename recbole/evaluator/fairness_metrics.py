
# -*- encoding: utf-8 -*-
# Fairness Metrics for Recommendation Systems

from logging import getLogger
import numpy as np
from scipy.spatial.distance import jensenshannon

from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType


class DemographicParityJSD(AbstractMetric):
    r"""DP-JSD (Demographic Parity - Jensen-Shannon Divergence) measures fairness
    across different sensitive attribute groups (e.g., gender, age).

    It computes the Jensen-Shannon divergence between the item recommendation
    distributions of different groups. Lower values indicate more fairness.

    .. math::
        \mathrm{DP-JSD@K} = JSD(P_{group1}(items), P_{group2}(items))

    where :math:`P_{group}(items)` is the distribution of recommended items for a group.

    Note:
        - DP-JSD = 0: Perfect fairness (identical distributions)
        - DP-JSD < 0.1: Very fair
        - DP-JSD < 0.3: Moderately fair
        - DP-JSD > 0.5: Unfair

        Requires 'sensitive_attribute' in the dataset (e.g., user gender).
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.sensitive_attribute", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get recommendation matrix, sensitive attributes, and number of items"""
        item_matrix = dataobject.get("rec.items").numpy()
        sensitive_attr = dataobject.get("data.sensitive_attribute")
        num_items = dataobject.get("data.num_items")
        return item_matrix, sensitive_attr, num_items

    def calculate_metric(self, dataobject):
        item_matrix, sensitive_attr, num_items = self.used_info(dataobject)

        # Check if sensitive attribute exists
        if sensitive_attr is None:
            self.logger.warning("Sensitive attribute not found. DP-JSD cannot be calculated.")
            return {}

        result = {}
        for k in self.topk:
            dp_jsd_value = self.compute_dp_jsd(item_matrix, sensitive_attr, num_items, k)
            key = "dpjsd@{}".format(k)
            # Handle NaN values
            if np.isnan(dp_jsd_value):
                result[key] = 0.0
            else:
                result[key] = round(dp_jsd_value, self.decimal_place)

        return result

    def compute_dp_jsd(self, item_matrix, sensitive_attr, num_items, topk):
        """Compute DP-JSD for top-k recommendations"""
        from scipy.spatial.distance import jensenshannon

        n_users = item_matrix.shape[0]

        # Create binary matrix for top-k items
        rank_topk_items = np.zeros((n_users, num_items), dtype=np.int8)
        user_indices = np.arange(n_users)[:, None]

        # Handle case where topk > item_matrix.shape[1]
        actual_k = min(topk, item_matrix.shape[1])
        rank_topk_items[user_indices, item_matrix[:, :actual_k]] = 1

        # Split by sensitive attribute
        unique_groups = np.unique(sensitive_attr)
        if len(unique_groups) < 2:
            return 0.0  # Cannot compute fairness with only one group

        # Use first two groups for binary comparison
        group1_mask = (sensitive_attr == unique_groups[0])
        group2_mask = (sensitive_attr == unique_groups[1])

        # Compute item distributions for each group
        group1_dist = np.sum(rank_topk_items[group1_mask], axis=0)
        group2_dist = np.sum(rank_topk_items[group2_mask], axis=0)

        # Normalize to probability distributions
        group1_dist = group1_dist / (group1_dist.sum() + 1e-10)
        group2_dist = group2_dist / (group2_dist.sum() + 1e-10)

        # Compute Jensen-Shannon divergence
        jsd = jensenshannon(group1_dist, group2_dist)

        return jsd


class EqualOpportunityJSD(AbstractMetric):
    r"""EOPP-JSD (Equal Opportunity - Jensen-Shannon Divergence) measures fairness
    in recommendation opportunities for relevant items across different groups.

    Unlike DP-JSD which considers all recommendations, EOPP-JSD only considers
    items that are actually relevant to users (ground truth items).

    .. math::
        \mathrm{EOPP-JSD@K} = JSD(P_{group1}(relevant\_items), P_{group2}(relevant\_items))

    Note:
        - EOPP-JSD = 0: Perfect equal opportunity
        - EOPP-JSD < 0.1: Very fair
        - EOPP-JSD < 0.3: Moderately fair
        - EOPP-JSD > 0.5: Unfair

        Requires 'sensitive_attribute' in the dataset.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "rec.topk", "data.sensitive_attribute", "data.num_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get recommendation matrix, ground truth, sensitive attributes, and number of items"""
        item_matrix = dataobject.get("rec.items").numpy()
        pos_matrix = dataobject.get("rec.topk").numpy()  # Ground truth relevant items
        sensitive_attr = dataobject.get("data.sensitive_attribute")
        num_items = dataobject.get("data.num_items")
        return item_matrix, pos_matrix, sensitive_attr, num_items

    def calculate_metric(self, dataobject):
        item_matrix, pos_matrix, sensitive_attr, num_items = self.used_info(dataobject)

        # Check if sensitive attribute exists
        if sensitive_attr is None:
            self.logger.warning("Sensitive attribute not found. EOPP-JSD cannot be calculated.")
            return {}

        result = {}
        for k in self.topk:
            eopp_jsd_value = self.compute_eopp_jsd(
                item_matrix, pos_matrix, sensitive_attr, num_items, k
            )
            key = "eoppjsd@{}".format(k)
            # Handle NaN values
            if np.isnan(eopp_jsd_value):
                result[key] = 0.0
            else:
                result[key] = round(eopp_jsd_value, self.decimal_place)

        return result

    def compute_eopp_jsd(self, item_matrix, pos_matrix, sensitive_attr, num_items, topk):
        """Compute EOPP-JSD for top-k recommendations"""
        from scipy.spatial.distance import jensenshannon

        n_users = item_matrix.shape[0]

        # Create binary matrices
        rank_topk_items = np.zeros((n_users, num_items), dtype=np.int32)
        truth_items = np.zeros((n_users, num_items), dtype=np.int32)

        user_indices = np.arange(n_users)[:, None]

        # Handle case where topk > matrix width
        actual_k = min(topk, item_matrix.shape[1])
        rank_topk_items[user_indices, item_matrix[:, :actual_k]] = 1

        # Mark ground truth items
        for i in range(n_users):
            truth_items[i, pos_matrix[i]] = 1

        # Only consider relevant items that were recommended
        truth_rank_items = truth_items & rank_topk_items

        # Split by sensitive attribute
        unique_groups = np.unique(sensitive_attr)
        if len(unique_groups) < 2:
            return 0.0

        group1_mask = (sensitive_attr == unique_groups[0])
        group2_mask = (sensitive_attr == unique_groups[1])

        # Compute relevant item distributions for each group
        group1_dist = np.sum(truth_rank_items[group1_mask], axis=0)
        group2_dist = np.sum(truth_rank_items[group2_mask], axis=0)

        # Normalize
        group1_dist = group1_dist / (group1_dist.sum() + 1e-10)
        group2_dist = group2_dist / (group2_dist.sum() + 1e-10)

        # Compute Jensen-Shannon divergence
        jsd = jensenshannon(group1_dist, group2_dist)

        return jsd
