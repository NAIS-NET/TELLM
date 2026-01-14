from .snapshot_utils import Read_Snapshot
from .cluster_utils import Cluster_Info
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import json
from pathlib import Path
import random

class TEDatasetWithinCluster(Dataset):
    def __init__(self, props, cluster, start, end, policy_align_mode=False):
        self.props = props
        self.cluster = cluster
        self.hist_len = props.hist_len
        self.list_tms = []
        self.list_tms_hist = []
        self.list_tms_pred = []
        self.list_capacities = []
        self.list_node_features = []
        self.list_node_features_norm = []
        self.list_optimal_values = []
        self.list_btl_edges = []
        self.dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"

        self.policy_align_mode = policy_align_mode

        self.results_path = f"{self.dataset_dir}/results/{self.props.topo}/{props.num_paths_per_pair}sp/{self.cluster}"
        filenames = np.loadtxt(f"{self.results_path}/filenames.txt", dtype="U", delimiter=",").reshape(-1, 3)
        filenames = filenames[start:end]
        
        if self.props.failure_id == None:
            file = open(f"{self.results_path}/optimal_values.txt")
            opts = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            opts = opts[start:end]

            # btl_edges = []
            # with open(f"{self.results_path}/bottleneck_edges.txt", 'r') as f:
            #     for line in f:
            #         btl_edges.append(json.loads(line.strip()))

        else:
            # NOTE, not supported for te llm currently
            file = open(f"{self.results_path}/optimal_values_failure_id_{self.props.failure_id}.txt")
            opts = np.loadtxt(file, dtype=np.float32).ravel()
            file.close()
            if len(opts) == 0:
                exit(1)

        for i, (snapshot_filename, opt_value) in enumerate(zip(filenames, opts)):
            topology_filename, pairs_filename, tm_filename = snapshot_filename
            self.snapshot = Read_Snapshot(self.props, topology_filename, pairs_filename, tm_filename, self.dataset_dir)
            # snapshot.tm [pair_demands * num_paths_per_pair, 1]
            self.list_tms.append(self.snapshot.tm)
            self.list_tms_pred.append(self.snapshot.tm_pred)
            self.list_optimal_values.append(opt_value)
            self.list_capacities.append(self.snapshot.capacities)
            self.list_node_features.append(self.snapshot._node_features)
            self.list_node_features_norm.append(self.snapshot._node_features_norm)
            # self.list_btl_edges.append(btl_info)

            # Generate history tm of shape [pair_demands * num_paths_per_pair, hist_len] with sliding window
            tms_hist = np.zeros((self.snapshot.tm.shape[0], self.hist_len))
            # Fill history with available previous snapshots, pad with zeros if not enough history
            for j in range(self.hist_len):
                hist_idx = i - self.hist_len + j + 1  # Calculate the index for history
                if hist_idx >= 0 and hist_idx < len(self.list_tms):
                    # Use available historical data
                    tms_hist[:, j] = self.list_tms[hist_idx].flatten()
                else:
                    # Pad with zeros when history is not available (beginning of dataset)
                    tms_hist[:, j] = 0
            
            self.list_tms_hist.append(tms_hist)
            
        cluster_info = Cluster_Info(self.snapshot, props, self.cluster, self.dataset_dir)
        self.cluster_info = cluster_info
        # self.edge_index = cluster_info.sp.get_edge_index().to(props.device)
        self.edge_index = cluster_info.sp.get_edge_index()
        self.pij = cluster_info.compute_ksp_paths(props.num_paths_per_pair, cluster_info.sp.pairs)
        self.pte = cluster_info.get_paths_to_edges_matrix(self.pij)
        self.padded_edge_ids_per_path = cluster_info.get_padded_edge_ids_per_path(self.pij, cluster_info.edges_map)
        self.num_pairs = cluster_info.num_pairs
        self.pairs = cluster_info.sp.pairs


    def __len__(self):
        return len(self.list_tms)
    
    def __getitem__(self, idx):
        if self.policy_align_mode:
            if random.random() < 0.5:
                policy_embeds = [0.0, 0.0, 0.0]
            else:
                policy_embeds = [
                    random.uniform(0, 1), 
                    random.uniform(0, 1), 
                    random.uniform(0, 1)
                    ]
        else:
            policy_embeds = [0.0, 0.0, 0.0]

        return {
            'node_features': self.list_node_features[idx],
            'node_features_norm': self.list_node_features_norm[idx],
            'capacities': self.list_capacities[idx],
            'tms': self.list_tms[idx],
            'tms_pred': self.list_tms_pred[idx] if self.props.pred else self.list_tms[idx],
            'tms_hist': self.list_tms_hist[idx],

            'optimal_values': self.list_optimal_values[idx],

            'edge_index': self.edge_index,
            'paths_to_edges': self.pte,
            'padded_edge_ids_per_path': self.padded_edge_ids_per_path,
            'policy_embeds': policy_embeds
        }
