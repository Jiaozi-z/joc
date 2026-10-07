import numpy as np
import scipy.sparse as sp
import torch
from collections import OrderedDict
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType,build_sim,build_knn_normalized_graph

import torch.nn.functional as F

import os

class FECA(GeneralRecommender):
    r"""FECA is a GCN-based recommender model.
    FECA enhances recommendation accuracy and diversity by integrating textual features of users and items
    with interaction information. The model learns user and item representations by linearly propagating
    embeddings on the user-item interaction graph and combines them with textual features to obtain the final
    embedding through weighted summation. FECA also incorporates edge embeddings and popularity features to
    further improve recommendation performance.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FECA, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_side_e = None
        self.restore_side_user_e = None
        self.restore_side_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e", "restore_side_e"]
        self.side_alpha = config['side_alpha'] if 'side_alpha' in config.final_config_dict else 0.3
        self.text_dim=config["text_dim"]

        self.inter_features = dataset.get_inter_feature().to(self.device)
        self.INTER_TEXT_FIELD = config["INTER_TEXT_FIELD"]
        self.query_embeddings=self.inter_features[config["INTER_TEXT_FIELD"]]
        self.query_linear = torch.nn.Linear(self.text_dim, self.latent_dim) 
        self.user_item_query_map = self.get_user_item_query_map() 

        self.item_features = dataset.get_item_feature().to(self.device)
        try:
            item_price_raw = self.item_features[config["ITEM_PRICE_FIELD"]]
            if hasattr(item_price_raw, 'float'):
                self.item_price = item_price_raw.float()
            else:
                self.item_price = torch.ones(self.n_items, device=self.device)
        except (AttributeError, KeyError):
            self.item_price = torch.ones(self.n_items, device=self.device)

        try:
            item_time_raw = self.item_features[config["ITEM_TIME_FIELD"]]
            if hasattr(item_time_raw, 'float'):
                self.item_time = item_time_raw.float()
            else:
                self.item_time = torch.ones(self.n_items, device=self.device)
        except (AttributeError, KeyError):
            self.item_time = torch.ones(self.n_items, device=self.device)

        self.price_control_baseline = config['price_control_baseline']
        self.time_control_baseline = config['time_control_baseline']

        self.alignment_loss_weight = config['alignment_loss_weight']
        self.exposure_penalty_weight = config['exposure_penalty_weight']

        self.enable_edge_embedding = config['enable_edge_embedding']

        # Ablation: disable text features entirely (MGF text branch)
        self.disable_text_features = config['disable_text_features'] if 'disable_text_features' in config.final_config_dict else False

        # PU-aware score adjustment
        self.enable_pu_score_adjustment = config['enable_pu_score_adjustment'] if 'enable_pu_score_adjustment' in config.final_config_dict else False
        self.pu_score_weight = config['pu_score_weight'] if 'pu_score_weight' in config.final_config_dict else 0.1
        self.pu_penalty_mode = config['pu_penalty_mode'] if 'pu_penalty_mode' in config.final_config_dict else 'both'
        self.pu_distribution = config['pu_distribution'] if 'pu_distribution' in config.final_config_dict else 'fixed'

        if self.pu_distribution in ('uniform', 'normal'):
            gen = torch.Generator()
            gen.manual_seed(2026)
            if self.pu_distribution == 'uniform':
                pu_t = 0.3 + torch.rand(self.n_users, 1, generator=gen) * 1.7
                tu_t = 0.3 + torch.rand(self.n_users, 1, generator=gen) * 1.7
            else:
                pu_t = (torch.randn(self.n_users, 1, generator=gen) * 0.3 + 1.0).clamp(0.3, 2.0)
                tu_t = (torch.randn(self.n_users, 1, generator=gen) * 0.3 + 1.0).clamp(0.3, 2.0)
            self.register_buffer('user_pu_thresh', pu_t)
            self.register_buffer('user_tu_thresh', tu_t)

        # Price/Time embedding weighting
        self.enable_pt_embedding = config['enable_pt_embedding'] if 'enable_pt_embedding' in config.final_config_dict else False
        self.pt_price_weight = config['pt_price_weight'] if 'pt_price_weight' in config.final_config_dict else 0.1
        self.pt_time_weight = config['pt_time_weight'] if 'pt_time_weight' in config.final_config_dict else 0.1
        self.pu_price_ratio = config['pu_price_ratio'] if 'pu_price_ratio' in config.final_config_dict else 1.0
        self.pu_time_ratio = config['pu_time_ratio'] if 'pu_time_ratio' in config.final_config_dict else 1.0

        # PU loss during training
        self.enable_pu_loss = config['enable_pu_loss'] if 'enable_pu_loss' in config.final_config_dict else False
        self.pu_loss_weight = config['pu_loss_weight'] if 'pu_loss_weight' in config.final_config_dict else 0.01

        # Tail boost: add score bonus to long-tail items at inference
        self.enable_tail_boost = config['enable_tail_boost'] if 'enable_tail_boost' in config.final_config_dict else False
        self.tail_boost_weight = config['tail_boost_weight'] if 'tail_boost_weight' in config.final_config_dict else 1.0
        self.tail_ratio = config['tail_ratio'] if 'tail_ratio' in config.final_config_dict else 0.75

        # Tail CL: extra BPR on tail pos items
        self.enable_tail_cl = config['enable_tail_cl'] if 'enable_tail_cl' in config.final_config_dict else False
        self.tail_cl_weight = config['tail_cl_weight'] if 'tail_cl_weight' in config.final_config_dict else 0.1

        # Inverse-popularity BPR weighting: upweight training samples where pos_item is tail
        self.enable_pop_inv_weight = config['enable_pop_inv_weight'] if 'enable_pop_inv_weight' in config.final_config_dict else False
        self.pop_inv_alpha = config['pop_inv_alpha'] if 'pop_inv_alpha' in config.final_config_dict else 1.0

        # Graph edge reweighting: amplify tail item edges in norm adj matrix
        self.enable_graph_tail_reweight = config['enable_graph_tail_reweight'] if 'enable_graph_tail_reweight' in config.final_config_dict else False
        self.graph_tail_factor = config['graph_tail_factor'] if 'graph_tail_factor' in config.final_config_dict else 2.0

        self.item_text_embeddings_origin = self.item_features[config["ITEM_TEXT_FIELD"]]
        try:
            self.item_text_embeddings = torch.nn.Embedding.from_pretrained(self.item_text_embeddings_origin, freeze=False)
        except (AttributeError, AssertionError):
            if hasattr(self.item_text_embeddings_origin, 'shape'):
                self.item_text_embeddings = torch.nn.Embedding.from_pretrained(self.item_text_embeddings_origin, freeze=False)
            else:
                self.item_text_embeddings = torch.nn.Embedding(self.n_items, self.text_dim)
        self.item_text_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_text_embeddings.weight.data)


        self.w_spe_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.rank_gate_network = torch.nn.Sequential(
            torch.nn.Linear(1, self.latent_dim),
            torch.nn.Sigmoid()
        )


        if hasattr(self.item_text_embeddings_origin, 'shape'):
            text_input_dim = self.item_text_embeddings_origin.shape[1]
        else:
            text_input_dim = self.text_dim
        self.text_trs = torch.nn.Linear(text_input_dim, self.latent_dim)
        self.gate_t = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Sigmoid()
        )
        self.query_common = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.latent_dim, 1, bias=False)
        )

        self.gate_text_prefer = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Sigmoid()
        )

        self.text_transform = torch.nn.Linear(self.latent_dim, self.latent_dim)


        self.softmax = torch.nn.Softmax(dim=-1)
        self.sparse = config["sparse"]
        self.cl_loss_weight  = config[
            "cl_loss_weight"
        ] 
        self.knn_k = config["knn_k"]
        self.temperature = config["temperature"]
        self.text_original_adj = self.build_text_original_adj().to(self.device)
        
        self.item_popularity = self.calculate_item_popularity()
        self.tail_mask = self._build_tail_mask()
        if self.enable_graph_tail_reweight:
            self.norm_adj_matrix = self._build_tail_reweighted_adj().to(self.device)

        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.Sigmoid()
        )

        self.item_representation_network = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU()
        )
        self.popularity_transform = torch.nn.Linear(1, self.latent_dim)

        self.popularity_network = torch.nn.Sequential(
            torch.nn.Linear(1, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.attribute_network = torch.nn.Sequential(
            torch.nn.Linear(text_input_dim, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.item_network = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + text_input_dim, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.alignment_loss = torch.nn.MSELoss()

        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map
        self.best_user_embeddings = None
        self.best_item_embeddings = None

        

    def get_original_user_id(self, remapped_id):
        return self.user_id_map.get(remapped_id.item(), remapped_id.item())

    def get_original_item_id(self, remapped_id):
        return self.item_id_map.get(remapped_id.item(), remapped_id.item())
 
    def build_text_original_adj(self):
        text_adj = build_sim(self.item_text_embeddings.weight.detach())
        text_adj = build_knn_normalized_graph(
            text_adj, 
            topk=self.knn_k,         
            is_sparse=self.sparse,    
            norm_type='sym'         
        )
        return text_adj

    def calculate_item_popularity(self):
        # 计算每个商品的流行度特征
        item_ids = self.inter_features[self.ITEM_ID].cpu().numpy()
        unique, counts = np.unique(item_ids, return_counts=True)
        exposure_count = np.zeros(len(self.item_features[self.ITEM_ID]))
        exposure_count[unique] = counts
        return torch.tensor(exposure_count, dtype=torch.float32).to(self.device)

    def _build_tail_mask(self):
        counts = self.item_popularity.cpu().numpy()
        sorted_counts = np.sort(counts)
        cut = max(int(len(sorted_counts) * self.tail_ratio), 1)
        threshold = sorted_counts[cut - 1]
        tail_mask = torch.tensor(counts <= threshold, dtype=torch.float32).to(self.device)
        return tail_mask

    def _build_tail_reweighted_adj(self):
        # Rebuild norm adj matrix with tail item edges amplified by graph_tail_factor
        inter_M = self.interaction_matrix
        inter_M_t = inter_M.transpose()
        tail_cpu = self.tail_mask.cpu().numpy()

        rows, cols, vals = [], [], []
        # user->item edges (item side amplified for tail items)
        for r, c in zip(inter_M.row, inter_M.col):
            factor = self.graph_tail_factor if tail_cpu[c] else 1.0
            rows.append(r)
            cols.append(c + self.n_users)
            vals.append(factor)
        # item->user edges
        for r, c in zip(inter_M_t.row, inter_M_t.col):
            factor = self.graph_tail_factor if tail_cpu[r] else 1.0
            rows.append(r + self.n_users)
            cols.append(c)
            vals.append(factor)

        n = self.n_users + self.n_items
        A = sp.coo_matrix((np.array(vals, dtype=np.float32),
                           (np.array(rows), np.array(cols))),
                          shape=(n, n)).todok()
        sumArr = np.array((A > 0).sum(axis=1)).flatten() + 1e-7
        diag = np.power(sumArr, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        self.R = L.tolil()[:self.n_users, self.n_users:]
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size(L.shape))


    def get_user_item_query_map(self):
        user_item_query_map = {}
        user_item_query_count = {}

        for user, item, query in zip(self.inter_features[self.USER_ID],
                                     self.inter_features[self.ITEM_ID],
                                     self.inter_features[self.INTER_TEXT_FIELD]):
            if isinstance(query, str):
                continue
            user_item_pair = (user.item(), item.item())
            if user_item_pair not in user_item_query_map:
                user_item_query_map[user_item_pair] = query
                user_item_query_count[user_item_pair] = 1
            else:
                user_item_query_map[user_item_pair] += query
                user_item_query_count[user_item_pair] += 1

        for user_item_pair in user_item_query_map:
            user_item_query_map[user_item_pair] /= user_item_query_count[user_item_pair]

        edge_embeddings = torch.zeros(self.interaction_matrix.nnz, self.text_dim, device=self.device)
        for idx, (user, item) in enumerate(zip(self.interaction_matrix.row, self.interaction_matrix.col)):
            user_item_pair = (user, item)
            if user_item_pair in user_item_query_map:
                edge_embeddings[idx] = user_item_query_map[user_item_pair]


        return user_item_query_map

    def get_norm_adj_mat(self):
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        indices = np.array(list(data_dict.keys()), dtype=np.int64)
        A = sp.coo_matrix(
            (
                np.ones(len(data_dict), dtype=np.float32),
                (indices[:, 0], indices[:, 1]),
            ),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        ).todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        self.R = L.tolil()[:self.n_users, self.n_users:]
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings,item_embeddings

    def forward(self, querys=None, user_indices=None, item_indices=None):
        user_embeddings, item_embeddings = self.get_ego_embeddings()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        popularity_embeddings = self.popularity_network(self.item_popularity.unsqueeze(1))
        
        item_embeddings_with_id = torch.cat([self.item_embedding.weight, self.item_text_embeddings.weight], dim=1)
        item_embeddings_with_id = self.item_network(item_embeddings_with_id)
        

        text_feats = self.text_trs(self.item_text_embeddings.weight)
        text_item_embeds = torch.multiply(item_embeddings, self.gate_t(text_feats))

        if self.disable_text_features:
            # Ablation: skip text KNN propagation and gate fusion, use zeros for side_embeds
            text_item_embeds = torch.zeros_like(text_item_embeds)

        if querys is not None and self.enable_edge_embedding:
            edge_embeddings = self.query_linear(querys)
            n_interactions = self.interaction_matrix.nnz
            full_edge_embeddings = torch.zeros(n_interactions, self.latent_dim, device=self.device)
            
            if user_indices is None or item_indices is None:
                batch_size = edge_embeddings.shape[0]
                full_edge_embeddings[:batch_size] = edge_embeddings
            else:
                for i, (user, item) in enumerate(zip(user_indices, item_indices)):
                    interaction_index = np.where((self.interaction_matrix.row == user.item()) & 
                                                (self.interaction_matrix.col == item.item()))[0]
                    if len(interaction_index) > 0:
                        full_edge_embeddings[interaction_index] = edge_embeddings[i]

            for layer_idx in range(self.n_layers):
                new_embeddings = self.propagate(embeddings_list[-1], full_edge_embeddings)
                embeddings_list.append(new_embeddings)

        else:
            for layer_idx in range(self.n_layers):
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
                embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        # Apply price/time weighting to item embeddings (only during inference)
        if self.enable_pt_embedding and not self.training:
            price_norm = torch.clamp(self.item_price / (self.item_price.mean() + 1e-8), 0.1, 10.0)
            time_norm = torch.clamp(self.item_time / (self.item_time.mean() + 1e-8), 0.1, 10.0)
            price_penalty = torch.log(1.0 + self.pt_price_weight * price_norm)
            time_penalty = torch.log(1.0 + self.pt_time_weight * time_norm)
            pt_weight = torch.exp(-(price_penalty + time_penalty)).unsqueeze(1)
            item_all_embeddings = item_all_embeddings * pt_weight

        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        content_embeds = lightgcn_all_embeddings

        att_common = self.query_common(text_embeds)
        weight_common = self.softmax(att_common)

        N, D = text_embeds.shape
        if N * N * D * 4 > 2 * 1024**3:
            # Memory-efficient path: avoids materializing (N,N,D) tensor
            # Equivalent to: unsqueeze broadcast → sep → gate → mean(dim=1)
            # Since softmax(dim=-1) on (N,1) = all 1.0, the result is
            # side_embeds = mean(text_embeds, dim=0) / 2  broadcast to (N, D)
            global_mean = text_embeds.mean(dim=0, keepdim=True)
            side_embeds = global_mean.expand(N, D) / 2
        else:
            common_embeds = weight_common.unsqueeze(dim=1) * text_embeds
            sep_text_embeds = text_embeds - common_embeds
            text_prefer = self.gate_text_prefer(content_embeds)
            sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
            side_embeds = (sep_text_embeds + common_embeds) / 2
            if side_embeds.dim() == 3:
                side_embeds = side_embeds.mean(dim=1)

        if self.training == False:
            self.final_user_embeddings = user_all_embeddings.detach().cpu()
            self.final_item_embeddings = item_all_embeddings.detach().cpu()
        
        if (user_indices is None or item_indices is None) and querys is not None:
            self.best_user_embeddings = user_all_embeddings
            self.best_item_embeddings = item_all_embeddings
            return user_all_embeddings, item_all_embeddings, side_embeds, content_embeds, popularity_embeddings, item_embeddings_with_id
        return user_all_embeddings, item_all_embeddings

    def propagate(self, embeddings, edge_embeddings):
        aggregated_embeddings = torch.sparse.mm(self.norm_adj_matrix, embeddings)
        
        # 考虑边嵌入的影响
        edge_influence = self.compute_edge_influence(embeddings, edge_embeddings)
        
        alpha=0.0001
        dropout_rate = 0.1
        updated_embeddings = F.relu(aggregated_embeddings) + F.relu(edge_influence * alpha)
        return updated_embeddings

    def compute_edge_influence(self, embeddings, edge_embeddings):
        user_indices = self.interaction_matrix.row
        item_indices = self.interaction_matrix.col + self.n_users
        
        user_influence = torch.zeros_like(embeddings)
        item_influence = torch.zeros_like(embeddings)
        
        user_influence.index_add_(0, torch.tensor(user_indices, device=self.device), edge_embeddings)
        item_influence.index_add_(0, torch.tensor(item_indices, device=self.device), edge_embeddings)
        
        user_degrees = torch.bincount(torch.tensor(user_indices, device=self.device), minlength=self.n_users).float() + 1e-7
        item_degrees = torch.bincount(torch.tensor(self.interaction_matrix.col, device=self.device), minlength=self.n_items).float() + 1e-7
        
        user_influence[:self.n_users] /= user_degrees.unsqueeze(1)
        item_influence[self.n_users:] /= item_degrees.unsqueeze(1)
        
        return user_influence + item_influence

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
            self.restore_side_user_e = None
            self.restore_side_item_e = None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        unique_pairs = OrderedDict()
        for idx, (u, i) in enumerate(zip(user, pos_item)):
            unique_pairs[(u.item(), i.item())] = idx

        batch_queries = []
        for user_item_pair in unique_pairs:
            if user_item_pair in self.user_item_query_map:
                batch_queries.append(self.user_item_query_map[user_item_pair])
            else:
                batch_queries.append(torch.zeros(self.text_dim, device=self.device))
        batch_queries = torch.stack(batch_queries)

        user_all_embeddings, item_all_embeddings, side_embeds, content_embeds, popularity_embeddings, item_embeddings_with_id = self.forward(batch_queries)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        side_user_e, side_item_e = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        if self.enable_pop_inv_weight:
            # Weight each BPR sample inversely by pos_item popularity: tail items get higher weight
            pop = self.item_popularity[pos_item].clamp(min=1.0)
            inv_weight = (1.0 / torch.log(pop + 1 + self.pop_inv_alpha)).detach()
            inv_weight = inv_weight / inv_weight.mean()  # normalize so mean weight = 1
            mf_loss = (inv_weight * torch.nn.functional.softplus(neg_scores - pos_scores)).mean()
        else:
            mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        side_embeds_items = side_embeds_items[pos_item]
        side_embeds_users = side_embeds_users[user]

        max_item_index = min(side_embeds_items.size(0), content_embeds_items.size(0)) - 1
        valid_pos_item = torch.clamp(pos_item, 0, max_item_index)

        max_user_index = min(side_embeds_users.size(0), content_embeds_user.size(0)) - 1
        valid_user = torch.clamp(user, 0, max_user_index)

        cl_loss = self.InfoNCE(side_embeds_items[valid_pos_item], content_embeds_items[valid_pos_item], self.temperature) + \
                self.InfoNCE(side_embeds_users[valid_user], content_embeds_user[valid_user], self.temperature)

        alignment_loss = self.alignment_loss(popularity_embeddings, item_embeddings_with_id)

        pos_item_exposure = self.item_popularity[pos_item]
        

        exposure_penalty = torch.log(pos_item_exposure + 1)
        exposure_loss = torch.mean(exposure_penalty * pos_scores)

        # PU loss: penalize high price and high time items
        pu_loss = torch.tensor(0.0, device=self.device)
        if self.enable_pu_loss:
            pos_price = self.item_price[pos_item]
            pos_time = self.item_time[pos_item]
            pu_penalty = torch.log(pos_price + 1.0) + torch.log(pos_time + 1.0)
            pu_loss = torch.mean(pu_penalty)

        # Tail loss: extra BPR on tail pos items to boost their scores
        tail_cl_loss = torch.tensor(0.0, device=self.device)
        if self.enable_tail_cl:
            is_tail = self.tail_mask[pos_item].bool()
            if is_tail.sum() > 0:
                tail_cl_loss = self.mf_loss(pos_scores[is_tail], neg_scores[is_tail])

        loss = mf_loss + self.reg_weight * reg_loss + self.cl_loss_weight * cl_loss + \
            self.alignment_loss_weight * alignment_loss + self.exposure_penalty_weight * exposure_loss + \
            self.pu_loss_weight * pu_loss + self.tail_cl_weight * tail_cl_loss

        print(f"mf_loss={mf_loss}, self.reg_weight * reg_loss={self.reg_weight * reg_loss}, "
            f"self.cl_loss_weight * cl_loss={self.cl_loss_weight * cl_loss}, "
            f"alignment_loss={ self.alignment_loss_weight *alignment_loss}, exposure_loss={self.exposure_penalty_weight * exposure_loss}, "
            f"pu_loss={self.pu_loss_weight * pu_loss}")
        print(f"loss={loss}")


        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_item_pair = (user.item(), item.item())
        if user_item_pair in self.user_item_query_map:
            query_embedding = self.user_item_query_map[user_item_pair].unsqueeze(0)
        else:
            query_embedding = torch.zeros(1, self.text_dim).to(self.device)

        user_all_embeddings, item_all_embeddings, *_ = self.forward(query_embedding, user, item)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def adjust_time_weights(self, average_time, time_control_baseline,diff_weight=2.247, time_scale=0.0861):
        time_diff_ratio = (average_time - time_control_baseline ) / average_time
        time_control_weight = torch.exp(1 + diff_weight*torch.tensor(time_diff_ratio)) * time_scale
        return time_control_weight
    
    def adjust_price_weights(self, average_price, price_control_baseline,diff_weight=3.70731, price_scale=0.001651):
        price_diff_ratio = (average_price - price_control_baseline ) / average_price
        price_control_weight = torch.exp(1 + diff_weight*torch.tensor(price_diff_ratio)) * price_scale
        return price_control_weight
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        if not hasattr(self, 'final_user_embeddings') or not hasattr(self, 'final_item_embeddings'):
            self.final_user_embeddings = self.restore_user_e.detach().cpu()
            self.final_item_embeddings = self.restore_item_e.detach().cpu()

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        item_prices = self.item_price.unsqueeze(0).expand(scores.shape[0], -1)
        item_times = self.item_time.unsqueeze(0).expand(scores.shape[0], -1)

        average_price = item_prices.mean().item()
        average_time = item_times.mean().item()

        if self.price_control_baseline>0:
            price_control_weight=self.adjust_price_weights(average_price,self.price_control_baseline)
        else:
            price_control_weight=0

        if self.time_control_baseline>0:
            time_control_weight=self.adjust_time_weights(average_time,self.time_control_baseline)
        else:
            time_control_weight=0


        price_diff = (item_prices - self.price_control_baseline) / self.price_control_baseline
        price_effect = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff)) * price_control_weight
        scores -= price_effect

        time_diff = (item_times - self.time_control_baseline) / self.time_control_baseline
        time_effect = torch.where(time_diff > 0, time_diff, torch.zeros_like(time_diff)) * time_control_weight
        scores -= time_effect

        # PU-aware score adjustment with per-user personalization
        if self.enable_pu_score_adjustment:
            safe_prices = torch.nan_to_num(item_prices, nan=0.0)
            safe_times = torch.nan_to_num(item_times, nan=0.0)
            price_norm = safe_prices / (safe_prices.nanmean() + 1e-8)
            time_norm = safe_times / (safe_times.nanmean() + 1e-8)

            w = self.pu_score_weight
            if self.pu_distribution in ('uniform', 'normal'):
                pu_thresh = self.user_pu_thresh[user].to(scores.device)
                tu_thresh = self.user_tu_thresh[user].to(scores.device)
                if self.pu_penalty_mode == 'time_only':
                    pu_penalty = w * torch.clamp(time_norm - tu_thresh, min=0.0)
                elif self.pu_penalty_mode == 'price_only':
                    pu_penalty = w * torch.clamp(price_norm - pu_thresh, min=0.0)
                else:
                    pu_penalty = w * (torch.clamp(price_norm - pu_thresh, min=0.0) + torch.clamp(time_norm - tu_thresh, min=0.0))
            else:
                if self.pu_penalty_mode == 'time_only':
                    pu_penalty = torch.log(time_norm + 1.0) * w
                elif self.pu_penalty_mode == 'price_only':
                    pu_penalty = torch.log(price_norm + 1.0) * w
                else:
                    pu_penalty = torch.log(self.pu_price_ratio * price_norm * self.pu_time_ratio * time_norm + 1.0) * w
            scores -= pu_penalty

        # Tail boost: only at final test, not during validation (boost_enabled set by trainer)
        if self.enable_tail_boost and getattr(self, 'boost_enabled', True):
            scores += self.tail_mask.unsqueeze(0) * self.tail_boost_weight

        return scores.view(-1)
    


    def get_query_embedding(self, user, item):
        user_item_pair = (user.item(), item.item())
        if user_item_pair in self.user_item_query_map:
            return self.user_item_query_map[user_item_pair].unsqueeze(0)
        else:
            return torch.zeros(1, self.text_dim).to(self.device)
        
    def get_user_item_embedding(self, user, item):
        user_emb = self.user_embedding.weight[user]
        item_emb = self.item_embedding.weight[item]
        return user_emb, item_emb

    
    # def save_final_embeddings(self, save_dir='./feca_saved_embeddings'):
    #     if hasattr(self, 'final_user_embeddings') and hasattr(self, 'final_item_embeddings'):
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, 'final_user_embeddings.npy'), self.final_user_embeddings.numpy())
    #         np.save(os.path.join(save_dir, 'final_item_embeddings.npy'), self.final_item_embeddings.numpy())
    #         print("final_embeddings have been saved to", save_dir)
    #     else:
    #         print("no valid final_embeddings to save")

    # def save_best_epoch_embeddings(self, save_dir='./feca_best_epoch_saved_embeddings'):
    #     if self.best_user_embeddings is not None and self.best_item_embeddings is not None:
    #         os.makedirs(save_dir, exist_ok=True)
    #         np.save(os.path.join(save_dir, 'best_user_embeddings.npy'), self.best_user_embeddings.detach().cpu().numpy())
    #         np.save(os.path.join(save_dir, 'best_item_embeddings.npy'), self.best_item_embeddings.detach().numpy())
    #         print("best_embeddings have been saved to", save_dir)
    #     else:
    #         print("no valid best_embeddings to save")