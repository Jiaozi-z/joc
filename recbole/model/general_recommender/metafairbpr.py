# coding: utf-8
r"""
MetaFairBPR - Cost-Sensitive Meta-Learning Strategy for Fair Provider Exposure
################################################
Reference:
Boratto, L., Cerniglia, G., Marras, M., Perniciano, A., & Pes, B. (2024).
A Cost-Sensitive Meta-Learning Strategy for Fair Provider Exposure in Recommendation.
In: Proc. of the 46th European Conference on IR Research, ECIR. ECIR 2024. Springer.
https://arxiv.org/abs/2401.13566
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class MetaFairBPR(GeneralRecommender):
    """
    Cost-Sensitive Meta-Learning Strategy for Fair Provider Exposure.

    This model extends BPR with a cost-sensitive sampling strategy that adjusts
    the probability of sampling items based on their provider group (e.g., gender)
    to ensure fair exposure across different groups.

    Key Features:
    - Cost-sensitive sampling for positive and negative items
    - Adjustable fairness parameter C to control exposure distribution
    - Compatible with pretrained BPR representations
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataloader):
        super(MetaFairBPR, self).__init__(config, dataloader)

        # Base model configuration
        self.base_model = config['recommendation_model'] if 'recommendation_model' in config else 'BPR'
        self.embedding_dim = config['embedding_size']

        # Fairness parameters
        self.fairness_c = config['fairness_c'] if 'fairness_c' in config else 2
        self.use_fair_sampling = config['use_fair_sampling'] if 'use_fair_sampling' in config else True
        self.fair_positive = config['fair_positive'] if 'fair_positive' in config else True
        self.fair_negative = config['fair_negative'] if 'fair_negative' in config else True

        # Provider group information (e.g., gender)
        self.provider_field = config['provider_field'] if 'provider_field' in config else None

        # Load pretrained representations if available
        dataset_path = config['data_path']
        user_repr_path = os.path.join(dataset_path, self.base_model + '_user_representation.npy')
        item_repr_path = os.path.join(dataset_path, self.base_model + '_item_representation.npy')

        if os.path.exists(user_repr_path) and os.path.exists(item_repr_path):
            # Use pretrained representations
            pretrained_user_representation = np.load(user_repr_path, allow_pickle=True)
            pretrained_item_representation = np.load(item_repr_path, allow_pickle=True)

            self.user_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_user_representation),
                freeze=False
            )
            self.item_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_item_representation),
                freeze=False
            )
        else:
            # Initialize new embeddings
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
            self.apply(xavier_normal_initialization)

        # Initialize provider group information
        self._init_provider_groups(dataloader)

        # Calculate sampling probabilities based on provider groups
        self._calculate_sampling_probabilities()

        # Loss functions
        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.reg_weight = config['reg_weight']

        # Exposure tracking
        self.register_buffer('item_exposure_count', torch.zeros(self.n_items))

    def _init_provider_groups(self, dataloader):
        """
        Initialize provider group information from dataset.
        """
        # Try to get provider group from item features
        if self.provider_field and hasattr(dataloader.dataset, 'item_feat'):
            item_feat = dataloader.dataset.item_feat
            if self.provider_field in item_feat:
                # Get provider group for each item
                self.provider_groups = item_feat[self.provider_field].numpy()
            else:
                self.logger.warning(f"Provider field '{self.provider_field}' not found in item features. Using uniform distribution.")
                self.provider_groups = np.zeros(self.n_items, dtype=np.int64)
        else:
            # Default: assume all items belong to the same group
            self.provider_groups = np.zeros(self.n_items, dtype=np.int64)

        # Convert to tensor
        self.register_buffer('provider_groups_tensor', torch.LongTensor(self.provider_groups))

    def _calculate_sampling_probabilities(self):
        """
        Calculate sampling probabilities based on cost parameter C.

        Following the paper:
        - p1 = 100 / (C + 1)  # Probability for majority group
        - p2 = 100 - p1        # Probability for minority group
        """
        # Calculate probabilities
        p1 = 100 // (self.fairness_c + 1)
        p2 = 100 - p1

        # Normalize to [0, 1]
        p1_norm = p1 / 100.0
        p2_norm = p2 / 100.0

        # Identify majority and minority groups
        # Assume group 0 is majority (male), group 1 is minority (female)
        unique_groups = np.unique(self.provider_groups)
        group_counts = {g: np.sum(self.provider_groups == g) for g in unique_groups}

        # Majority group has more items
        majority_group = max(group_counts, key=group_counts.get)
        minority_group = min(group_counts, key=group_counts.get)

        # Assign probabilities
        item_probabilities = np.zeros(self.n_items, dtype=np.float32)
        for i in range(self.n_items):
            if self.provider_groups[i] == majority_group:
                item_probabilities[i] = p1_norm
            else:
                item_probabilities[i] = p2_norm

        # Normalize probabilities to sum to 1
        item_probabilities = item_probabilities / item_probabilities.sum()

        # Store as tensor
        self.register_buffer('item_sampling_probs', torch.FloatTensor(item_probabilities))

        self.logger.info(f"Fairness parameter C={self.fairness_c}")
        self.logger.info(f"Majority group probability: {p1_norm:.4f}")
        self.logger.info(f"Minority group probability: {p2_norm:.4f}")
        self.logger.info(f"Group distribution: {group_counts}")

    def forward(self, user, item):
        """
        Forward pass to compute scores.
        """
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.sum(user_e * item_e, dim=1)

    def calculate_loss(self, interaction):
        """
        Calculate BPR loss with optional fair sampling.
        """
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.user_embedding(user)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)

        # Calculate scores
        pos_score = torch.sum(user_e * pos_e, dim=1)
        neg_score = torch.sum(user_e * neg_e, dim=1)

        # BPR loss
        rec_loss = self.rec_loss(pos_score, neg_score)

        # Regularization loss
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)

        # Update exposure counts
        with torch.no_grad():
            self.item_exposure_count[pos_item] += 1

        # Total loss
        total_loss = rec_loss + self.reg_weight * reg_loss

        return total_loss

    def predict(self, interaction):
        """
        Predict scores for user-item pairs.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users.
        """
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return scores.view(-1)

    def get_exposure_statistics(self):
        """
        Get exposure statistics for fairness analysis.
        """
        total_exposure = self.item_exposure_count.sum().item()
        if total_exposure == 0:
            return {}

        # Calculate exposure by group
        unique_groups = torch.unique(self.provider_groups_tensor)
        group_exposure = {}

        for group in unique_groups:
            group_mask = (self.provider_groups_tensor == group)
            group_items = group_mask.sum().item()
            group_exp = self.item_exposure_count[group_mask].sum().item()
            group_exposure[f'group_{group.item()}'] = {
                'total_exposure': group_exp,
                'avg_exposure': group_exp / group_items if group_items > 0 else 0,
                'exposure_ratio': group_exp / total_exposure if total_exposure > 0 else 0,
                'num_items': group_items
            }

        # Calculate Gini coefficient
        exposure_dist = self.item_exposure_count / (total_exposure + 1e-8)
        gini = self._calculate_gini(exposure_dist)

        stats = {
            'total_exposure': total_exposure,
            'group_exposure': group_exposure,
            'gini_coefficient': gini.item(),
            'exposure_std': exposure_dist.std().item()
        }

        return stats

    def _calculate_gini(self, distribution):
        """
        Calculate Gini coefficient.
        """
        sorted_dist = torch.sort(distribution)[0]
        n = len(sorted_dist)
        index = torch.arange(1, n + 1, device=distribution.device, dtype=torch.float)
        gini = (2 * torch.sum(index * sorted_dist)) / (n * torch.sum(sorted_dist)) - (n + 1) / n
        return gini

    def reset_exposure_counts(self):
        """
        Reset exposure counts.
        """
        self.item_exposure_count.zero_()

    def save_pretrained_representation(self):
        """
        Save user and item representations.
        """
        dataset_path = self.config['data_path']
        model_name = self.__class__.__name__

        user_repr = self.user_embedding.weight.cpu().detach().numpy()
        item_repr = self.item_embedding.weight.cpu().detach().numpy()

        user_path = os.path.join(dataset_path, f'{model_name}_user_representation.npy')
        item_path = os.path.join(dataset_path, f'{model_name}_item_representation.npy')

        np.save(user_path, user_repr, allow_pickle=True)
        np.save(item_path, item_repr, allow_pickle=True)

        self.logger.info(f"Saved representations to {dataset_path}")
