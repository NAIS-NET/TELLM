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
from src.utils import print_config, logging, display_result, setup_rich_logging  
from rich.progress import Progress

from src.policy_layer.agent import chat 
setup_rich_logging()

agent_config= OmegaConf.load('configs/chat.yaml')

output = chat(
    'We are launching a new app!',
    api_key=agent_config.language_model.api_key,
    base_url=agent_config.language_model.base_url,
    model=agent_config.language_model.model_name,
)

logging.info(f"Extracted policy embeddings: {output.policy_embeddings}")
logging.info(f"Chatbot response:\n{output.response}")

ckpt_path = 'outputs/tellm_moe'

dataset="abilene"
gpu_id = 6
num_for_loops = 5

test_start = 238
test_end = 278
MODEL_ID = os.path.join(ckpt_path, "pretrained")

pipe = TeLLMInferencePipline(MODEL_ID, device=f"cuda:{gpu_id}", torch_dtype=torch.float32)

policy_embeds=output.policy_embeddings

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

dynamic_12_config = {
    "topo": "dynamic_12",           # Name of the topology to be used.
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

abilene_config = {
    "topo": "abilene",           # Name of the topology to be used.
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

b4_config = {
    "topo": "b4",           # Name of the topology to be used.
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

geant_config = {
    "topo": "geant",           # Name of the topology to be used.
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


kdl_config = {
    "topo": "kdl",           # Name of the topology to be used.
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

g50_config = {
    "topo": "g50",           # Name of the topology to be used.
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

configs = {
    'dynamic_12': dynamic_12_config,
    'kdl': kdl_config,
    'b4': b4_config,
    'geant': geant_config,
    'abilene': abilene_config,
    'g50': g50_config,
    'brain': brain_config,
}

dataset_cfg = OmegaConf.create(configs[dataset])
test_dataset = TEDatasetWithinCluster(dataset_cfg, cluster=0, start=test_start, end=test_end)

progress = Progress()
all_losses = []
with progress:
    task = progress.add_task("Testing...", total=test_end-test_start)
    for i in range(test_end-test_start):
        # Correct indexing by adding test_start
        sample_idx = i
        out = pipe(test_dataset[sample_idx], policy_embeds=policy_embeds, num_for_loops=num_for_loops,return_loss=True)
        
        # Update progress description with current loss
        progress.update(task, advance=1, description=f"norm_mlu: {out.loss.norm_mlu.item():.4f}")
        
        all_losses.append(out.loss)


average_losses = {}
for key in all_losses[0].keys():
    average_losses[key] = sum(loss_dict[key] for loss_dict in all_losses) / len(all_losses)


display_result(average_losses)

