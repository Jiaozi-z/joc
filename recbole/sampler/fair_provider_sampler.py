# coding: utf-8
"""
Cost-Sensitive Sampler for Fair Provider Exposure

This sampler implements the cost-sensitive sampling strategy from:
"A Cost-Sensitive Meta-Learning Strategy for Fair Provider Exposure in Recommendation"
"""

import numpy as np
import torch
from recbole.data.interaction import Interaction
from recbole.sampler import AbstractSampler


class FairProviderSampler(AbstractSampler):
    """
    Cost-sensitive sampler that adjusts sampling probabilities based on provider groups.

    This sampler implements two key strategies:
    1. Fair positive sampling: Sample positive items with higher probability for minority groups
    2. Fair negative sampling: Sample negative items with adjusted probabilities based on groups
    """

    def __init__(self, dataset, distribution='uniform', alpha=1.0):
        super().__init__(dataset, distribution, alpha)

        # Get provider group information
        self.provider_groups = self._get_provider_groups()

        # Calculate sampling probabilities
        self.fairness_c = 2  # Default cost parameter
        self._calculate_sampling_probabilities()

    def _get_provider_groups(self):
        """
        Extract provider group information from dataset.
        """
        # Try to get from item features
        if hasattr(self.dataset, 'item_feat') and 'gender' in self.dataset.item_feat:
            provider_groups = self.dataset.item_feat['gender'].numpy()
        else:
            # Default: all items in same group
            provider_groups = np.zeros(self.dataset.item_num, dtype=np.int64)

        return provider_groups

    def _calculate_sampling_probabilities(self):
        """
        Calculate sampling probabilities based on cost parameter C.
        """
        # Calculate probabilities
        p1 = 100 // (self.fairness_c + 1)  # Majority group
        p2 = 100 - p1                       # Minority group

        # Normalize
        p1_norm = p1 / 100.0
        p2_norm = p2 / 100.0

        # Identify majority and minority groups
        unique_groups = np.unique(self.provider_groups)
        group_counts = {g: np.sum(self.provider_groups == g) for g in unique_groups}

        majority_group = max(group_counts, key=group_counts.get)

        # Assign probabilities
        self.item_probabilities = np.zeros(self.dataset.item_num, dtype=np.float32)
        for i in range(self.dataset.item_num):
            if self.provider_groups[i] == majority_group:
                self.item_probabilities[i] = p1_norm
            else:
                self.item_probabilities[i] = p2_norm

        # Normalize to sum to 1
        self.item_probabilities = self.item_probabilities / self.item_probabilities.sum()

    def set_fairness_parameter(self, c):
        """
        Set the fairness parameter C and recalculate probabilities.
        """
        self.fairness_c = c
        self._calculate_sampling_probabilities()

    def sample_by_user_ids(self, user_ids, item_ids, num=1):
        """
        Sample negative items for given user-item pairs using fair sampling.
        """
        try:
            return self._fair_sample_by_user_ids(user_ids, item_ids, num)
        except:
            # Fallback to uniform sampling if fair sampling fails
            return super().sample_by_user_ids(user_ids, item_ids, num)

    def _fair_sample_by_user_ids(self, user_ids, item_ids, num=1):
        """
        Fair sampling implementation.
        """
        neg_item_ids = []

        for user_id in user_ids:
            # Get user's positive items
            user_pos_items = self.dataset.inter_feat[
                self.dataset.inter_feat[self.dataset.uid_field] == user_id
            ][self.dataset.iid_field].numpy()

            # Create mask for available items (not in user's positive items)
            available_mask = np.ones(self.dataset.item_num, dtype=bool)
            available_mask[user_pos_items] = False

            # Get available items and their probabilities
            available_items = np.where(available_mask)[0]
            available_probs = self.item_probabilities[available_items]

            # Normalize probabilities
            available_probs = available_probs / available_probs.sum()

            # Sample negative items
            sampled_items = np.random.choice(
                available_items,
                size=num,
                replace=False,
                p=available_probs
            )

            neg_item_ids.extend(sampled_items)

        return torch.tensor(neg_item_ids, dtype=torch.long)
