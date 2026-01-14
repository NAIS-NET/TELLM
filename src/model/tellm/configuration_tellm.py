#!/usr/bin/env python3
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging 

logger = logging.get_logger(__name__)


class TeLLMSolverConfig(PretrainedConfig):
    model_type = "tellm"
    def __init__(
        self,
        policy_hidden_size=3,
        num_gnn_layers=1,
        num_transformer_layers=1,
        dropout=0.,
        num_mlp1_hidden_layers=2,
        num_mlp2_hidden_layers=2,
        num_heads=0,
        num_paths_per_pair=4,
        hist_len=12,
        mag_decouple=True,
        decoder="default",
        num_experts=4,
        moe_top_k=2,
        noisy_gating=True,
        noise_epsilon=1e-2,
        **kwargs
    ):
        self.policy_hidden_size = policy_hidden_size
        self.num_gnn_layers = num_gnn_layers
        self.num_transformer_layers = num_transformer_layers
        self.dropout = dropout
        self.num_mlp1_hidden_layers = num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = num_mlp2_hidden_layers
        self.num_heads = num_heads
        self.num_paths_per_pair = num_paths_per_pair
        self.hist_len = hist_len
        self.mag_decouple = mag_decouple

        # MoE
        self.decoder = decoder # moe or default
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon

        super().__init__(**kwargs)

__all__ = ["TeLLMSolverConfig"]
