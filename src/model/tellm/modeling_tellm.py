#!/usr/bin/env python3
from transformers.utils import add_start_docstrings, logging
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput


import torch_scatter
from transformers import ModernBertConfig, AutoModel

from .configuration_tellm import TeLLMSolverConfig
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

logger = logging.get_logger(__name__)

# apply monkey patch for non-contiguous attn tensor error
import transformers.models.modernbert.modeling_modernbert as modernbert_module
from transformers.models.modernbert.modeling_modernbert import MODERNBERT_ATTENTION_FUNCTION
from .flash_attn_monkey_patch import flash_attention_forward as reshape_flash_attention_forward
MODERNBERT_ATTENTION_FUNCTION["flash_attention_2"] = reshape_flash_attention_forward
logger.info("Applied monkey patch for flash attention 2 in ModernBertAttention to replace view with reshape. (Fixes non-contiguous attn tensor error)")



@dataclass
class TeMetrics(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlu: Optional[torch.FloatTensor] = None
    norm_mlu: Optional[torch.FloatTensor] = None
    optimal: Optional[torch.FloatTensor] = None
    history_term: Optional[torch.FloatTensor] = None
    global_term: Optional[torch.FloatTensor] = None
    cost_term: Optional[torch.FloatTensor] = None
    max_sensitivity: Optional[torch.FloatTensor] = None


@dataclass
class TeOutput(ModelOutput):
    """Output dataclass for the path solver."""
    gammas: Optional[torch.FloatTensor] = None
    edges_util: Optional[torch.FloatTensor] = None
    split_ratios: Optional[torch.FloatTensor] = None
    loss: Optional[TeMetrics] = None


class TeLLMSolverPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models."""
    config_class = TeLLMSolverConfig
    base_model_prefix = "tellm"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

TeLLM_START_DOCSTRING = r"""
This version of TeLLM takes advantages of MoE expressiveness, HARP iterative decoding techniques, Figret history robustness, our online policy alignment techniques,
and magnitude decoupling techniques.
"""


# class FigretNetWork(nn.Module):
#     def __init__(self, input_dim, output_dim, layer_num):
#         """Initialize the FigretNetWork with the network structure.

#         Args:
#             input_dim: dimension of input data, history len * flattened traffic matrix
#             output_dim: dimension of output data, len of candidate paths all s-d pairs
#             layer_num: number of hidden layers
#         """
#         super(FigretNetWork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.layers = []
#         self.layers.append(nn.Linear(input_dim, 128))
#         self.layers.append(nn.ReLU())
#         for _ in range(layer_num):
#             self.layers.append(nn.Linear(128, 128))
#             self.layers.append(nn.ReLU())
#         self.layers.append(nn.Linear(128, output_dim))
#         # self.layers.append(nn.Sigmoid())
#         self.net = nn.Sequential(*self.layers)
    
#     def forward(self, x):
#         """Forward the input data through the network.

#         Args:
#             x: input data, history len * flattened traffic matrix
#         """
#         x = self.flatten(x)
#         logits = self.net(x)
#         return logits
    # def _initialize_weights(self, method='xavier'):
    #         """Initialize network weights with specified method"""
    #         for module in self.modules():
    #             if isinstance(module, nn.Linear):
    #                 if method == 'xavier':
    #                     nn.init.xavier_uniform_(module.weight)
    #                 elif method == 'xavier_normal':
    #                     nn.init.xavier_normal_(module.weight)
    #                 elif method == 'he':
    #                     nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    #                 elif method == 'he_normal':
    #                     nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    #                 elif method == 'normal':
    #                     nn.init.normal_(module.weight, mean=0, std=0.2)
    #                 elif method == 'uniform':
    #                     nn.init.uniform_(module.weight, -0.1, 0.1)
                    
    #                 # 偏置初始化
    #                 if module.bias is not None:
    #                     nn.init.constant_(module.bias, 0)


@add_start_docstrings(
    TeLLM_START_DOCSTRING,
)
class TeLLMSolverModel(TeLLMSolverPreTrainedModel):
    def __init__(self, config: TeLLMSolverConfig):
        super().__init__(config)
        self.config = config
        self.mag_decouple = config.mag_decouple

        if config.decoder == 'moe':
            self.solver = TEHeadMoE(config)
        else:
            self.solver = TEHead(config) 

        self.post_init()


    def forward(
            self, 
            te_inputs, 
            policy_embeds=None,
            optimal_mlu=None,
            return_loss=False,
            num_for_loops=3,
            num_paths_per_pair=4,
            hist_alpha_base=500,
            global_alpha_base=5,
            cost_alpha_base=1,
            ):
        batch_size, num_pairs, _ = te_inputs['tms_hist'].shape
        if policy_embeds is None:
            policy_embeds = torch.zeros((batch_size, self.policy_hidden_size), device=te_inputs['tms'].device)
        policy_embeds = policy_embeds.unsqueeze(1).repeat(1, num_pairs, 1)

        capacities = te_inputs['capacities']
        tms = te_inputs['tms']
        tms_hist = te_inputs['tms_hist']

        if self.mag_decouple:
            # normalization for generalization across topos
            capacities_max, _ = capacities.max(dim=1, keepdim=True)
            capacities_norm = capacities / capacities_max

            tms_max, _ = tms.max(dim=1, keepdim=True)
            tms_norm = tms / tms_max


            tms_hist_norm = tms_hist / tms_max

            te_inputs['capacities_raw'] = capacities
            te_inputs['tms_raw'] = tms
            te_inputs['tms_hist_raw'] = tms_hist

            mag_ratio = tms_max / capacities_max
            mag_ratio = mag_ratio.repeat(1, num_pairs, 1)
            mag_ratio = mag_ratio.clamp(max=1.0)

            te_inputs['capacities'] = capacities_norm
            te_inputs['tms'] = tms_norm
            te_inputs['tms_hist'] = tms_hist_norm
            te_inputs['mag_ratio'] = mag_ratio
            te_inputs['node_features'] = te_inputs['node_features_norm']
        else:
            te_inputs['capacities_raw'] = capacities
            te_inputs['tms_raw'] = tms
            te_inputs['tms_hist_raw'] = tms_hist
        
        # gammas [batch_size, num_tunnels]
        gammas, edges_util = self.solver(**te_inputs, policy_embeds=policy_embeds, num_for_loops=num_for_loops, num_paths_per_pair=num_paths_per_pair)

        # ####
        # tms_hist = te_inputs['tms_hist'] # shape: (batch_size, hist_len, num_nodes * (num_nodes - 1))
        # batch_size = tms_hist.shape[0]
        # tms_hist = tms_hist[:, ::num_paths_per_pair, :].contiguous()
        # tms_hist.view(batch_size, -1) # shape: (batch_size, hist_len * num_nodes * (num_nodes - 1))
        # gammas = self.solver_b(tms_hist)
        # [batch_size, num_pairs, num_paths_per_pair]
        gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
        split_ratios = torch.nn.functional.softmax(gammas, dim=-1)
    
        if return_loss:
            loss = self.loss(te_inputs, 
            split_ratios, 
            edges_util, 
            policy_embeds, 
            optimal_mlu,
            hist_alpha_base,
            global_alpha_base,
            cost_alpha_base,
            num_paths_per_pair
            )

        return TeOutput(
            gammas=gammas,
            edges_util=edges_util,
            split_ratios=split_ratios,
            loss=loss if return_loss else None
        )

    def loss(self, 
             te_inputs, 
             split_ratios, 
             edges_util, 
             policy_embeds, 
             optimal_mlu=None,
             hist_alpha_base=500,
             global_alpha_base=5,
             cost_alpha_base=1,
             num_paths_per_pair=4,
             ):
        """Compute the loss.
        """
        # NOTE edges_util is the hypernet utils

        # tm = te_inputs['tms'].to(torch.float32)
        # capacities = te_inputs['capacities'].to(torch.float32)
        split_ratios = split_ratios.to(torch.float32)
        tm_raw = te_inputs['tms_raw'].to(torch.float32)
        tm_hist = te_inputs['tms_hist'].to(torch.float32)
        capacities_raw = te_inputs['capacities_raw'].to(torch.float32)
        paths_to_edges = te_inputs['paths_to_edges'].to(torch.float32)

        policy_embeds = policy_embeds.to(torch.float32)
        num_demands = tm_raw.shape[1]
        assert torch.allclose(torch.sum(split_ratios, dim=2), torch.ones_like(torch.sum(split_ratios, dim=2)), rtol=1e-5, atol=1e-5), "split_ratios are not sum to 1"


        # assert alpha is same for a batch
        alpha_hist = policy_embeds[:, 0, 0] * hist_alpha_base
        alpha_global = policy_embeds[:, 0, 1] * global_alpha_base
        alpha_cost = policy_embeds[:, 0, 2] * cost_alpha_base

        # alpha_hist = policy_embeds[:, 0, 0] 
        # alpha_global = policy_embeds[:, 0, 1] 
        # alpha_cost = policy_embeds[:, 0, 2] 

        batch_size = split_ratios.shape[0]

        # Process paths and compute split ratios
        # num_demands x num_edges
        paths_to_edges = paths_to_edges.coalesce()

        split_ratios = split_ratios.reshape(batch_size, -1)

        # Compute link utilization directly from split ratios: batch_size x num_edges
        # NOTE this is equivalent to edges_util * mag_ratio
        edges_util_raw = self._compute_link_util(split_ratios, tm_raw, paths_to_edges, capacities_raw)

        # Get max values for each sample in the batch
        max_cong, _ = torch.max(edges_util, dim=1)  # Shape: (batch_size,)
        max_cong_raw, _ = torch.max(edges_util_raw, dim=1)

        # Compute robustness terms using shared function
        hist_term, max_sensitivity = self._compute_hist_term(split_ratios, tm_hist, batch_size, alpha=alpha_hist, num_paths_per_pair=num_paths_per_pair)
        global_term, _ = self._compute_global_term(split_ratios, batch_size, alpha=alpha_global)
        cost_term = self._compute_cost_term(split_ratios, batch_size, alpha=alpha_cost)

        # For loss: 1.0 - max_cong if max_cong == 0.0 else max_cong 
        zero_mask = (max_cong == 0.0)
        base_loss = torch.where(zero_mask, 1.0 - max_cong, max_cong)
        if not self.mag_decouple:
            base_loss = base_loss/base_loss.detach()
        losses = base_loss + hist_term + global_term + cost_term

        # Handle optimal_mlu processing
        if optimal_mlu is not None:
            optimal_mlu = optimal_mlu.to(torch.float32)
            # Validation loss (normalized by optimal MLU)
            norm_mlu = torch.where(zero_mask, 1.0 - max_cong_raw, max_cong_raw / optimal_mlu)
            
            mask = (norm_mlu < 0.999)
            if mask.any():
                print("Warning: Some validation losses are less than 1.0")
                print("max_cong[mask]:", max_cong_raw[mask])
                print("optimal_mlu[mask]:", optimal_mlu[mask])
                print("norm_mlu[mask]:", norm_mlu[mask])
            
            avg_norm_mlu = torch.mean(norm_mlu)
            opt = torch.mean(optimal_mlu)
        else:
            # When optimal_mlu is not provided, set both to None
            avg_norm_mlu = None
            opt = None

        # Compute averages
        loss = torch.mean(losses)
        mlu = torch.mean(max_cong)
        mlu_raw = torch.mean(max_cong_raw)
        hist_term_mean = torch.mean(hist_term)
        global_term_mean = torch.mean(global_term)
        cost_term_mean = torch.mean(cost_term)
        avg_max_sensitivity = torch.mean(max_sensitivity)  
        
        return TeMetrics(
            loss=loss,
            mlu=mlu_raw,
            norm_mlu=avg_norm_mlu,
            optimal=opt,
            history_term=hist_term_mean,
            global_term=global_term_mean,
            cost_term=cost_term_mean,
            max_sensitivity=avg_max_sensitivity
        )

    
    def _compute_link_util(self, split_ratios, tm, paths_to_edges, capacities):
        """
        Compute link utilization from split ratios in one step.
        
        Args:
            split_ratios: Traffic split ratios across paths
            tm: Traffic matrix
            paths_to_edges: Mapping from paths to edges
            capacities: Link capacities
            
        Returns:
            edges_util: Link utilization values
        """
        # Compute split demands
        split_demands = split_ratios * tm.squeeze(-1)
        
        # Compute link loads
        data_on_links = torch.sparse.mm(
            paths_to_edges.to(dtype=torch.float32).t(), 
            split_demands.to(dtype=torch.float32).t()
        ).t()
        
        # Compute link utilization
        edges_util = data_on_links / capacities
        
        # Handle special cases
        inf_mask = torch.where(edges_util == float('inf'))
        nan_mask = torch.isnan(edges_util)
        edges_util[inf_mask] = 1000
        edges_util[nan_mask] = 0
        
        return edges_util

    def _compute_hist_term(self, split_ratios, tm_hist, batch_size, alpha, num_paths_per_pair):
        """Compute robustness term for given traffic matrix history.
        
        Args:
            split_ratios: split ratios tensor (batch_size, num_demands * num_paths_per_pair)
            tm_hist: historical
            batch_size: batch size
            
        Returns:
            robustness_term: robustness term (batch_size,)
            max_sensitivity: maximum sensitivity per demand (batch_size, num_demands)
        """
        # Reshape split ratios: (batch_size, num_commodities, num_paths_per_commodity)
        split_ratios_reshaped = split_ratios.reshape(batch_size, -1, num_paths_per_pair)
        
        # Get max sensitivity per demand
        max_sensitivity = torch.max(split_ratios_reshaped, dim=2)[0]  # Shape: (batch_size, num_demands)
        
        # Repeat for each path
        max_sensitivity_repeated = max_sensitivity.repeat_interleave(
            num_paths_per_pair, 
            dim=1
        )
        
        # Compute standard deviation of historical traffic
        tm_hist_std = torch.std(tm_hist, dim=-1)  # Shape: (batch_size, num_demands)
        hist_nan_mask = torch.isnan(tm_hist_std)
        tm_hist_std[hist_nan_mask] = 0.0  # Handle NaN values by setting them to zero
        
        # Weight the sensitivity and compute mean
        weight_max_sensitivity = max_sensitivity_repeated * tm_hist_std  # Shape: (batch_size, num_demands)
        robustness_term = torch.mean(weight_max_sensitivity, dim=1)  # Shape: (batch_size,)
        robustness_term = alpha * robustness_term/robustness_term.detach() # Shape: (batch_size,)

        return robustness_term, max_sensitivity

    def _compute_global_term(self, split_ratios, batch_size, alpha):
        # Reshape split ratios: (batch_size, num_commodities, num_paths_per_commodity)
        split_ratios_reshaped = split_ratios.reshape(batch_size, -1, self.config.num_paths_per_pair)
        
        # Get max sensitivity per demand
        max_sensitivity = torch.max(split_ratios_reshaped, dim=2)[0]  # Shape: (batch_size, num_demands)
        
        # Repeat for each path
        max_sensitivity_repeated = max_sensitivity.repeat_interleave(
            self.config.num_paths_per_pair, 
            dim=1
        )
        
        global_term = torch.mean(max_sensitivity_repeated, dim=1)  # Shape: (batch_size,)
        global_term = alpha * global_term/global_term.detach() # Shape: (batch_size,)
        
        return global_term, max_sensitivity

    def _compute_cost_term(self, split_ratios, batch_size, alpha):
        """Compute robustness term for given traffic matrix history with path-order weighting.
        """
        # Reshape split ratios: (batch_size, num_commodities, num_paths_per_commodity)
        split_ratios_reshaped = split_ratios.reshape(batch_size, -1, self.config.num_paths_per_pair)

        # Create path order weights: higher weights for later paths when alpha > 1
        path_indices = torch.arange(self.config.num_paths_per_pair, device=split_ratios.device)
        path_weights = torch.pow(2, path_indices)  # Shape: (num_paths_per_pair,)
        # Apply weights to split ratios
        # Expand path_weights to match split_ratios_reshaped dimensions
        path_weights = path_weights.view(1, 1, -1).expand_as(split_ratios_reshaped)
        weighted_split_ratios = split_ratios_reshaped * path_weights  # Shape: (batch_size, num_demands, num_paths_per_pair)

        # Sum weighted split ratios for each demand
        weighted_cost = torch.sum(weighted_split_ratios, dim=2)  # Shape: (batch_size, num_demands)
        cost_term = torch.mean(weighted_cost, dim=1)  # Shape: (batch_size,)
        cost_term = alpha * cost_term/cost_term.detach() # Shape: (batch_size,)

        return cost_term


class RAUMoE(nn.Module):
    """Mixture of Experts module for replacing mlp2"""
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 num_mlp_hidden_layers, 
                 num_experts=4, 
                 top_k=2, 
                 noisy_gating: bool = True,
                 noise_epsilon: float = 1e-2
                 ):
        super(RAUMoE, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mlp_hidden_layers = num_mlp_hidden_layers

        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon
        
        # Create multiple expert networks
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.ModuleList()
            expert.append(nn.Linear(input_dim, input_dim))
            for _ in range(num_mlp_hidden_layers):
                expert.append(nn.Linear(input_dim, input_dim))
            expert.append(nn.Linear(input_dim, output_dim))
            self.experts.append(expert)
        
        # Gating network
        # self.gate = nn.Linear(input_dim, num_experts)
        self.gate = nn.Linear(3, num_experts)
        
        
    # generate delta_gammas at each round
    # inputs:  dnn_2_inputs bsz, numdemands x numpaths, hiddensize
    # outputs: delta_gammas bsz, numdemands x numpaths, 1
    def forward(self, x):
        batch_size, num_pairs, feature_dim = x.shape
        x_gate_flat = x[:,:,-3:].view(-1, 3)
        x_flat = x.view(-1, feature_dim)
        
        # Get gating weights
        gates = self.gate(x_gate_flat) # bsz*num_pairs x input_dim

        if self.noisy_gating and self.training:
            # Add noise for exploration during training
            noise = torch.randn_like(gates) * self.noise_epsilon
            gates = gates + noise

        # Select top-k experts
        topk_gate_weights, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        # gate_weights and mask
        topk_gate_weights = torch.softmax(topk_gate_weights, dim=-1)
        mask = torch.zeros_like(gates).scatter_(-1, top_k_indices, topk_gate_weights)

        # Initialize output, will be restored to bsz x numdemands x num_paths when return
        final_output = torch.zeros(batch_size * num_pairs, self.output_dim, 
                           device=x.device, dtype=x.dtype)
        
        # Process through selected experts
        for i, expert in enumerate(self.experts):
            expert_mask = mask[:, i] > 0
            if not expert_mask.any():
                continue
            
            # Get samples assigned to this expert
            expert_input = x_flat[expert_mask]
            expert_weight = mask[expert_mask, i:i+1]  # Keep dimension for broadcasting
            
            # Forward through expert layers
            for layer_idx, layer in enumerate(expert):
                if layer_idx == 0:
                    expert_output = layer(expert_input)
                    expert_output = expert_output.relu()
                elif layer_idx == self.num_mlp_hidden_layers + 1:
                    # Last layer, no activation
                    expert_output = layer(expert_output)
                else:
                    expert_output = layer(expert_output)
                    expert_output = expert_output.relu()
            
            # Weight by gating weight and accumulate
            final_output[expert_mask] += expert_weight * expert_output
        
        # Reshape back to original dimensions
        final_output = final_output.view(batch_size, num_pairs, self.output_dim)
        
        return final_output


# ModernBert as SetTransformer to enable flash-attention 2
class TransformerModel(nn.Module):
    def __init__(self, in_dim: int, nhead: int, dim_feedforward: int,
                 nlayers: int, dropout: float = 0.0, activation="gelu"):
        """
        Hugging Face ModernBert-based Transformer Encoder.
        """
        super().__init__()
        # Adjust based on your GPU memory, KDL scale problem might exceed block constrain of attention without chunking
        self.chunk_size = 65535  
        # Map to ModernBert config
        config = ModernBertConfig(
            hidden_size=in_dim,
            num_attention_heads=nhead,
            intermediate_size=dim_feedforward,
            num_hidden_layers=nlayers,
            attention_dropout=dropout,
            mlp_dropout=dropout,
            embedding_dropout=dropout,
            hidden_activation=activation,
        )
        
        self.transformer_encoder = AutoModel.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.in_dim = in_dim
        
        # ModernBert uses rotary positional embeddings (RoPE)
        # Set the RoPE theta values to extremely large values to effectively disable them
        # This makes positional information negligible, equivalent to set transformer
        config.global_rope_theta = 1e12  # Effectively disables global RoPE
        config.local_rope_theta = 1e12   # Effectively disables local RoPE

    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Process large batches in smaller chunks to avoid memory issues
        """
        batch_size = src.shape[0]
        
        if batch_size <= self.chunk_size:
            # Normal processing for small batches
            attention_mask = None
            if src_key_padding_mask is not None:
                attention_mask = src_key_padding_mask.int()
            
            outputs = self.transformer_encoder(
                inputs_embeds=src,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs.last_hidden_state
        else:
            # Process in chunks for large batches
            results = []
            for i in range(0, batch_size, self.chunk_size):
                end_idx = min(i + self.chunk_size, batch_size)
                chunk_src = src[i:end_idx]
                chunk_mask = src_key_padding_mask[i:end_idx] if src_key_padding_mask is not None else None
                
                attention_mask = chunk_mask.int() if chunk_mask is not None else None
                
                chunk_outputs = self.transformer_encoder(
                    inputs_embeds=chunk_src,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                results.append(chunk_outputs.last_hidden_state)
            
            return torch.cat(results, dim=0)


# GNN of HARP
class GNN(nn.Module):
    def __init__(self, num_features, num_gnn_layers):
        super(GNN, self).__init__()
        self.num_features = num_features
                
        self.gnns = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnns.append(GCNConv(num_features, num_features+1))
            elif i == 1:
                self.gnns.append(GCNConv(num_features+1, num_features+2))
            else:
                self.gnns.append(GCNConv(num_features+2, num_features+2))
        self.output_dim = num_gnn_layers*(self.num_features+2) - 1
        
        
        
    def forward(self, node_features, edge_index, capacities):
        """
        Forward pass of the GNN model.

        Args:
            node_features (torch.Tensor): Node features for each graph in the batch, 
                                        shape (batch_size, num_nodes, num_features).
            edge_index (torch.Tensor): Edge indices defining the graph connectivity, 
                                    shape (2, num_edges).
            capacities (torch.Tensor): Edge capacities for each graph in the batch, 
                                    shape (batch_size, num_edges).

        Returns:
            torch.Tensor: Edge embeddings for each edge in the graph, with capacities included, 
                        shape (batch_size, num_edges, output_dim).

        Process:
            1. Iterate over the batch of node features and capacities.
            2. For each graph in the batch:
                a. Pass the node features through each GNN layer (GCNConv), applying Leaky ReLU activation.
                b. Collect the intermediate node embeddings after each GNN layer.
                c. Concatenate node embeddings from all GNN layers if more than one GNN layer is used.
            3. Stack the node embeddings for all graphs in the batch.
            4. Expand edge_index to match the batch size, so it can be used for batch processing.
            5. Use the expanded edge_index to extract edge embeddings from the node embeddings.
            6. Sum the node embeddings corresponding to each edge and concatenate them with the edge capacities.
        """


        batch_size = node_features.shape[0]
        ne_list = []
        for i in range(batch_size):
            nf = node_features[i]
            caps = capacities[i]
            sample_ne_list = []
            for j, gnn in enumerate(self.gnns):
                if j == 0:
                    ne = gnn(nf, edge_index=edge_index, edge_weight=caps)
                else:
                    ne = gnn(ne, edge_index=edge_index, edge_weight=caps)
                ne = F.leaky_relu(ne, 0.02)
                sample_ne_list.append(ne)
            if len(self.gnns) > 1:
                node_embeddings = torch.cat((sample_ne_list), dim=-1)
                ne_list.append(node_embeddings)
            else:
                ne_list = sample_ne_list
        node_embeddings = torch.stack(ne_list).contiguous()
        edge_index_expanded = edge_index.t().expand(batch_size, -1, -1)
        batch_size, num_nodes, feature_size = node_embeddings.shape
        _, num_edges, _ = edge_index_expanded.shape
        
        # Create a batch index
        batch_index = torch.arange(batch_size).view(-1, 1, 1)
        batch_index = batch_index.repeat(1, num_edges, 2)  # Repeat the batch index for each edge
        edge_embeddings = node_embeddings[batch_index, edge_index_expanded]
        capacities = capacities.unsqueeze(-1)
        edge_embeddings = edge_embeddings.sum(dim=-2)
        edge_embeddings = torch.cat((edge_embeddings, capacities), dim=-1)
        
        return edge_embeddings


class TEHead(nn.Module):
    def __init__(self, config):

        super(TEHead, self).__init__()

        # Define the architecture of HARP
        self.num_gnn_layers = config.num_gnn_layers
        self.num_transformer_layers = config.num_transformer_layers
        self.dropout = config.dropout
        self.num_mlp1_hidden_layers = config.num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = config.num_mlp2_hidden_layers
        self.policy_hidden_size = config.policy_hidden_size

        # self.num_paths_per_pair = config.num_paths_per_pair
        self.hist_len = config.hist_len
        self.mag_decouple = config.mag_decouple

        # Define the GNN
        self.gnn = GNN(2, self.num_gnn_layers)

        self.input_dim = self.gnn.output_dim + 1
        
        # CLS Token for the Set Transformer
        self.cls_token = nn.Parameter(torch.Tensor(1, self.input_dim))
        nn.init.kaiming_normal_(self.cls_token, nonlinearity='relu')
        
        if config.num_heads == 0:
            num_heads = self.input_dim//4
        else:
            num_heads = config.num_heads
        
        # Define the Set Transformer
        self.transformer = TransformerModel(in_dim = self.input_dim, nhead=num_heads,
                            dim_feedforward=self.input_dim, nlayers=self.num_transformer_layers, 
                            dropout=self.dropout, activation="gelu")
        
        # Define the 1st MLP
        self.mlp_1_dim = self.input_dim + self.hist_len + self.policy_hidden_size  
        if self.mag_decouple:
            self.mlp_1_dim = self.mlp_1_dim + 1

        self.mlp1 = nn.ModuleList()
        self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))
        for i in range(self.num_mlp1_hidden_layers):
            self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))
        self.mlp1.append(nn.Linear(self.mlp_1_dim, 1))
        
        # Define the 2nd MLP (Recurrent Adjustment Unit - RAU)
        self.mlp_2_dim = self.input_dim + 2 + self.hist_len + self.policy_hidden_size
        if self.mag_decouple: # additional dim to keep relative magnitude info
            self.mlp_2_dim = self.mlp_2_dim + 1
        self.mlp2 = nn.ModuleList()
        self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        for i in range(self.num_mlp2_hidden_layers):
            self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        self.mlp2.append(nn.Linear(self.mlp_2_dim, 1))
        
        
    def forward(self, 
                policy_embeds, 
                node_features, 
                edge_index, 
                capacities, 
                padded_edge_ids_per_path,
                tms, 
                tms_hist, 
                paths_to_edges, 
                mag_ratio=None, 
                num_for_loops=3, 
                num_paths_per_pair=4,
                dynamic=True, 
                **kwargs):
        """
        NOTE: this module is modified from HARP, adding our normalized formulation, history information according to
        figret
            Process:
            1. Pass the node features, edge index, and capacities through the GNN to obtain edge embeddings.
            2. Expand the edge embeddings using the padded edge IDs per path.
            3. Add a CLS token to the edge embeddings and apply masking for attention.
            4. At this point, tunnels are described as a set of edges (edge embeddings)
            5. Pass the tunnels as sets of edges through the Set Transformer.
            6. Concatenate the transformer output for path embeddings (corresponds to the CLS token) with the predicted traffic matrix.
            7. Compute initial split ratios using the first MLP (mlp1).
            8. Perform iterative adjustments of split ratios using the second MLP (RAU) within the 
            specified number of for-loops. MLP2 takes as input (per tunnel):
                i) Demand of the pair that the tunnels is associated with
                ii) Network-wide MLU
                iii) Bottleneck link utilization in the tunnel
                iv) Tunnel embeddings conditioned on the bottleneck link as generated by the Set Transformer 
        """
        batch_size, num_pairs, _ = tms_hist.shape

        edge_embeddings_with_caps = self.gnn(node_features, edge_index, capacities)
        batch_size = tms.shape[0]
        total_number_of_paths = paths_to_edges.shape[0]

        edge_embeddings_with_caps = [edge_embeddings_with_caps[i][padded_edge_ids_per_path] \
               for i in range(edge_embeddings_with_caps.shape[0])]        
        edge_embeddings_with_caps = torch.stack(edge_embeddings_with_caps)
        cls_token = self.cls_token.unsqueeze(0)
        
        # If the topology changes across time (dynamic), then probably every example has a unique edge_embeddings_with_caps
        if dynamic:
            edge_embeddings_with_caps = torch.cat((cls_token.repeat(batch_size, paths_to_edges.shape[0], 1).unsqueeze(-2),
                                                    edge_embeddings_with_caps), dim=-2)
        else:
            edge_embeddings_with_caps = torch.cat((cls_token.repeat(1, paths_to_edges.shape[0], 1).unsqueeze(-2), 
                                                   edge_embeddings_with_caps), dim=-2)


        # If the topology does not change across examples/snapshots (static topology), just make one edge_embeddings_with_caps because it is the same for all examples

        attention_mask = torch.cat((torch.ones((padded_edge_ids_per_path.shape[0], 1), dtype=torch.bool, device=edge_embeddings_with_caps.device),
                                    (padded_edge_ids_per_path != -1.0)), dim=1)
        
        ori_dtype = edge_embeddings_with_caps.dtype
        attention_dtype = next(self.transformer.parameters()).dtype
        edge_embeddings_with_caps = edge_embeddings_with_caps.to(dtype=attention_dtype)

        # If the topology changes across examples/snapshots (dynamic topology), then feed each edge_embeddings_with_caps to the transformer
        if dynamic:
            out_trf_list = [self.transformer(edge_embeddings_with_caps[i, :, :, :].contiguous(), attention_mask) for i in range(batch_size)]
        
        # If the topology does not change across examples/snapshots (static topology), then feed one edge_embeddings_with_caps to the transformer
        else: # static topology
            out_trf_list = [self.transformer(edge_embeddings_with_caps[i, :, :, :].contiguous(), attention_mask) for i in range(1)]
        

        out_trf_list = torch.stack(out_trf_list)

        # If the topology does not change across examples/snapshots (static topology), then make batch_size copies of the transformer output, each belongs to one example/snapshot.
        if not dynamic:
            out_trf_list = out_trf_list.repeat(batch_size, 1, 1, 1)

        out_trf_list = out_trf_list.to(dtype=ori_dtype)

        
        path_embeddings = out_trf_list[:, :, 0, :]
        path_edge_embeddings = out_trf_list[:, :, 1:, :]
        
        # Predicted matrix        
        # path_embeddings = torch.cat((path_embeddings, tms_pred), dim=-1)
        path_embeddings = torch.cat((path_embeddings, tms_hist, policy_embeds), dim=-1)
        if self.mag_decouple:
            assert mag_ratio is not None
            path_embeddings = torch.cat((path_embeddings, mag_ratio), dim=-1)
        
        # Compute initial raw split ratios        
        for index, layer in enumerate(self.mlp1):
            if index == 0:
                gammas = layer(path_embeddings)
                gammas = gammas.relu()
            elif index == self.num_mlp1_hidden_layers + 1:
                gammas = layer(gammas)
            else:
                gammas = layer(gammas)
                gammas = gammas.relu()
        
        paths_to_edges = paths_to_edges.coalesce()
        indices = paths_to_edges.indices()
        values = paths_to_edges.values()
        row_indices = indices[0]
        col_indices = indices[1]
        
        for i in range(num_for_loops):
            if i > 0:
                gammas = new_gammas
            gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
            split_ratios = torch.nn.functional.softmax(gammas, dim=-1).reshape(batch_size, -1)
            
            split_ratios = split_ratios*tms.squeeze(-1)
                        
            # data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios.to(dtype=torch.float32).t()).t()
            dtype = split_ratios.dtype
            data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios.to(dtype=torch.float32).t()).t()
            data_on_links.to(dtype=dtype)
                        
            edges_util = data_on_links/capacities + 1e-4

            inf_mask = torch.where(edges_util == float('inf'))
            nan_mask = torch.isnan(edges_util)
            edges_util[inf_mask] = 1000 + 1e-4
            edges_util[nan_mask] = 0 + 1e-4
            
            mlu, mlu_indices = torch.max(edges_util, dim=-1)
            mlu -= 1e-4
            mlu = mlu.view(batch_size, 1, 1).repeat(1, total_number_of_paths, 1)
                        
            
            max_utilization_per_path, max_indices = torch_scatter.scatter_max((edges_util[:, col_indices] * values),
                                                                              row_indices, dim=1, dim_size=paths_to_edges.shape[0])
            max_indices = col_indices[max_indices]
            max_utilization_per_path -= 1e-4
            
            max_indices_expanded = max_indices.unsqueeze(2).expand(-1, -1,  padded_edge_ids_per_path.size(1))
            matches = (max_indices_expanded == padded_edge_ids_per_path)
            
            positions = matches.nonzero()
            positions = positions.view(batch_size, total_number_of_paths, -1)
            
            dim0_range = positions[:, :, 0].view(batch_size, total_number_of_paths, -1)
            dim1_range = positions[:, :, 1].view(batch_size, total_number_of_paths, -1)
            positions = positions[:, :, -1].view(batch_size, total_number_of_paths, -1)
            
            bottleneck_path_edge_embeddings = (path_edge_embeddings[dim0_range, dim1_range, positions]).squeeze(-2)
                                                
            dnn_2_inputs = torch.cat((bottleneck_path_edge_embeddings, 
                                      max_utilization_per_path.unsqueeze(-1),
                                      mlu,
                                      tms_hist,
                                      policy_embeds,
                                      ), dim=-1).to(dtype=dtype)
            if self.mag_decouple:
                dnn_2_inputs = torch.cat((dnn_2_inputs, mag_ratio), dim=-1)

            # # dnn2 inference
            # for index, layer in enumerate(self.mlp2):
            #     if index == 0:
            #         delta_gammas = layer(dnn_2_inputs)
            #         delta_gammas = delta_gammas.relu()
            #     elif index == self.num_mlp2_hidden_layers + 1:
            #         delta_gammas = layer(delta_gammas)
            #     else:
            #         delta_gammas = layer(delta_gammas)
            #         # batch_size, -1, 1
            #         delta_gammas = delta_gammas.relu()
            delta_gammas = self.iterative_refine(dnn_2_inputs)
                    
            # batch_size, numdemands, numpaths
            gammas = gammas.reshape(batch_size, -1, 1)
            new_gammas = delta_gammas + gammas
        
        if num_for_loops == 0:
            new_gammas = gammas
        new_gammas = new_gammas.reshape(batch_size, -1, num_paths_per_pair)
        split_ratios = torch.nn.functional.softmax(new_gammas, dim=-1)
        split_ratios = split_ratios.reshape(batch_size, -1) 
        
        # Actual matrix
        split_ratios = split_ratios*tms.squeeze(-1)
        data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), split_ratios.to(dtype=torch.float32).t()).t()
                
        edges_util = data_on_links/capacities
        inf_mask = torch.where(edges_util == float('inf'))
        nan_mask = torch.isnan(edges_util)
        edges_util[inf_mask] = 1000
        edges_util[nan_mask] = 0
        
        return new_gammas, edges_util

    def iterative_refine(self, dnn_2_inputs):
        # dnn2 inference
        for index, layer in enumerate(self.mlp2):
            if index == 0:
                delta_gammas = layer(dnn_2_inputs)
                delta_gammas = delta_gammas.relu()
            elif index == self.num_mlp2_hidden_layers + 1:
                delta_gammas = layer(delta_gammas)
            else:
                delta_gammas = layer(delta_gammas)
                # batch_size, -1, 1
                delta_gammas = delta_gammas.relu()
        return delta_gammas

class TEHeadMoE(TEHead):
    def __init__(self, config):
        super(TEHeadMoE, self).__init__(config)
        
        # MoE specific parameters
        self.num_experts = getattr(config, 'num_experts', 4)
        self.moe_top_k = getattr(config, 'moe_top_k', 2)
        self.noisy_gating = getattr(config, 'noisy_gating', True)
        self.noise_epsilon = getattr(config, 'noise_epsilon', 1e-2)
        
        # Replace mlp2 with MoE
        del self.mlp2  # Remove the original mlp2
        
        self.moe = RAUMoE(
            input_dim=self.mlp_2_dim,
            output_dim=1,
            num_experts=self.num_experts,
            num_mlp_hidden_layers=self.num_mlp2_hidden_layers,
            top_k=self.moe_top_k,
            noisy_gating=self.noisy_gating,
            noise_epsilon=self.noise_epsilon
        )
    
    def load_pretrained_weights(self, pretrained_model, freeze_non_moe=True):
        """
        Load weights from a pretrained TeLLMHarp model.
        
        Args:
            pretrained_model: Pretrained TeLLMHarp model or state dict
            freeze_non_moe: If True, freeze all non-MoE parameters
        """
        if isinstance(pretrained_model, dict):
            # It's a state dict
            state_dict = pretrained_model
            # Load all weights except mlp2
            filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('mlp2')}
            self.load_state_dict(filtered_dict, strict=False)
            
            # Initialize MoE experts with pretrained mlp2 weights if available
            mlp2_dict = {k.replace('mlp2.', ''): v for k, v in state_dict.items() if k.startswith('mlp2')}
            if mlp2_dict:
                for expert in self.moe.experts:
                    for i, layer in enumerate(expert):
                        if f'{i}.weight' in mlp2_dict:
                            layer.weight.data = mlp2_dict[f'{i}.weight'].clone()
                        if f'{i}.bias' in mlp2_dict:
                            layer.bias.data = mlp2_dict[f'{i}.bias'].clone()
        else:
            # It's a model
            # Copy all parameters except mlp2
            self.gnn.load_state_dict(pretrained_model.gnn.state_dict())
            self.transformer.load_state_dict(pretrained_model.transformer.state_dict())
            self.cls_token.data = pretrained_model.cls_token.data.clone()
            for i, layer in enumerate(self.mlp1):
                layer.load_state_dict(pretrained_model.mlp1[i].state_dict())
            
            # Initialize MoE experts with pretrained mlp2 weights
            for expert in self.moe.experts:
                for i, layer in enumerate(expert):
                    layer.load_state_dict(pretrained_model.mlp2[i].state_dict())
        
        # Optionally freeze non-MoE parameters for efficient fine-tuning
        if freeze_non_moe:
            for param in self.gnn.parameters():
                param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            for layer in self.mlp1:
                for param in layer.parameters():
                    param.requires_grad = False
            # Freeze the first expert in MoE (keep as baseline/reference)
            for param in self.moe.experts[0].parameters():
                param.requires_grad = False

    def iterative_refine(self, dnn_2_inputs):
        # dnn2 inference
        delta_gammas = self.moe(dnn_2_inputs)
        return delta_gammas


__all__ = ["TeLLMSolverModel", "TeLLMSolverPreTrainedModel", "TEHeadMoE"]
