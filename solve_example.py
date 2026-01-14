import os
from matplotlib import pyplot as plt
import networkx as nx
import test
import torch
from omegaconf import OmegaConf
from transformers import AutoModel
from src.model.tellm.modeling_tellm import TeLLMSolverModel
from src.model.tellm.configuration_tellm import TeLLMSolverConfig
from src.model.tellm.pipline_tellm import TeLLMInferencePipline
from tellm_main import TEDatasetWithinCluster
from src.utils import print_config, logging, display_result  
from rich.progress import track
from rich.progress import Progress


ckpt_path = 'outputs/tellm_moe'

dataset="abilene"
gpu_id = 1
num_paths_per_pair=4

num_for_loops = 5

test_start = 8000
test_end = 9000
if dataset == 'kdl':
    test_start = 238
    test_end = 278
elif dataset == 'g50':
    test_start = 240
    test_end = 280
elif dataset == 'facebook_pod_a':
    test_start = 7400
    test_end = 10000
else:
    test_start = 9000
    test_end = 10000


# test_start = 200
# test_end = 288
# MODEL_ID = os.path.join(ckpt_path, "pretrained")
MODEL_ID = os.path.join(ckpt_path, "pretrained")

pipe = TeLLMInferencePipline(MODEL_ID, device=f"cuda:{gpu_id}", torch_dtype=torch.float32)
# pipe.model.solver.moe.top_k = 2
# print(pipe.model.solver.moe_top_k) 

policy_embeds=[0.0, 0.0, 0.0]

# config_path = os.path.join(ckpt_path, "outputs", "config.yaml")
# config= OmegaConf.load(config_path)
# print_config(config, resolve=True)

brain_config = {
    "topo": "brain",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": 4,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}

facebook_pod_a_config = {
    "topo": "facebook_pod_a",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}


facebook_pod_b_config = {
    "topo": "facebook_pod_b",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}

abilene_config = {
    "topo": "abilene",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}

b4_config = {
    "topo": "b4",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}

geant_config = {
    "topo": "geant",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}


kdl_config = {
    "topo": "kdl",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}

g50_config = {
    "topo": "g50",           # Name of the topology to be used.
    "weight": None,              # Metric used for edge weights (optional).
    "metric": "MLU",             # Optimization metric (currently supports only 'MLU').
    "num_paths_per_pair": num_paths_per_pair,     # Number of paths per node pair.
    "framework": "harp",         # Optimization framework to use.
    "failure_id": None,          # Failure scenario ID (optional).
    "dynamic": True,             # Whether the topology is dynamic.
    "mode": "train",
    "pred": False,               # Use predicted TMs.
    "hist_len": 12               # Length of TM history to input.
}

configs = {
    'facebook_pod_a': facebook_pod_a_config,
    'facebook_pod_b': facebook_pod_b_config,
    'kdl': kdl_config,
    'b4': b4_config,
    'geant': geant_config,
    'abilene': abilene_config,
    'g50': g50_config,
    'brain': brain_config
}

dataset_cfg = OmegaConf.create(configs[dataset])
test_dataset = TEDatasetWithinCluster(dataset_cfg, cluster=0, start=test_start, end=test_end)
all_cost = []

progress = Progress()
all_losses = []
with progress:
    task = progress.add_task("Testing...", total=test_end-test_start)
    for i in range(test_end-test_start):
        # Correct indexing by adding test_start
        sample_idx = i
        out = pipe(test_dataset[sample_idx], policy_embeds=policy_embeds, num_for_loops=num_for_loops, num_paths_per_pair=num_paths_per_pair, return_loss=True)
        
        # Update progress description with current loss
        progress.update(task, advance=1, description=f"norm_mlu: {out.loss.norm_mlu.item():.4f}")

        split_ratios = out.split_ratios.squeeze(0).cpu()
        split_ratios_flat = split_ratios.reshape(-1)
        num_hops_per_path = test_dataset[sample_idx]['paths_to_edges'].sum(dim=1)

        total_weighted_hops = (num_hops_per_path * split_ratios_flat).sum()
        all_cost.append(total_weighted_hops)
        
        all_losses.append(out.loss)


average_losses = {}
for key in all_losses[0].keys():
    average_losses[key] = sum(loss_dict[key] for loss_dict in all_losses) / len(all_losses)

average_losses['average_cost'] = sum(all_cost) / len(all_cost)

display_result(average_losses)

