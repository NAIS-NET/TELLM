import json
import sys
import os
import pickle
import math


dataset_dir = 'dataset' 
dataset_dir = os.path.abspath(dataset_dir)

from src.data_utils.snapshot_utils import Read_Snapshot
from src.data_utils.cluster_utils import Cluster_Info

from gurobipy import GRB, Model
import gurobipy as gp
import numpy as np
import sys
import copy
import tqdm
from scipy.sparse import csr_matrix

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import print_config, logging


@hydra.main(config_path="configs", config_name="gurobi_mlu.yaml", version_base="1.2.0")
def main(props: OmegaConf):
    os.makedirs(f"{dataset_dir}/results", exist_ok=True)
    # props = OmegaConf.load("configs/gurobi_mlu.yaml")

    print_config(props)

    topo = props.topo
    num_paths_per_pair = props.num_paths_per_pair
    start_index = props.opt_start_idx
    if props.opt_end_idx > 0:
        end_index = props.opt_end_idx
    else:
        end_index = None

    results_path = f"{dataset_dir}/results/{topo}/{num_paths_per_pair}sp"


    os.makedirs(results_path, exist_ok=True)


    file_manifest = f"{dataset_dir}/manifest/{props.topo}_manifest.txt"
    manifest = np.loadtxt(file_manifest, dtype="U", delimiter=",")


    num_cluster  = 0

    os.makedirs(f"{dataset_dir}/{num_cluster}", exist_ok=True)
    os.makedirs(f"{dataset_dir}/topologies/paths_dict", exist_ok=True)
    os.makedirs(f"{dataset_dir}/topologies/paths", exist_ok=True)

    topology_filename, pairs_filename, tm_filename = manifest[start_index]

    topology_filename = topology_filename.strip()
    pairs_filename = pairs_filename.strip()
    tm_filename = tm_filename.strip()

    previous_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename, dataset_dir)
    current_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename, dataset_dir)

    cluster_info = Cluster_Info(current_sp, props, num_cluster, dataset_dir)
    cluster_info.pij = cluster_info.compute_ksp_paths(num_paths_per_pair, cluster_info.sp.pairs)
    cluster_info.paths_to_edges = csr_matrix(cluster_info.get_paths_to_edges_matrix(cluster_info.pij).to_dense().numpy())

    num_snapshots_in_cluster = 0

    if props.failure_id == None:
        cluster_path = f"{results_path}/{num_cluster}"
        os.makedirs(cluster_path, exist_ok=True)

        optimal_path = f"{results_path}/{num_cluster}/optimal_values.txt"

        optimal_values = open(optimal_path, "w")

        filenames_path = f"{results_path}/{num_cluster}/filenames.txt"
        filenames = open(filenames_path, "w")

        # HACK for bottleneck edges
        btl_path = f"{results_path}/{num_cluster}/bottleneck_edges.txt"
        bottleneck_edges = open(btl_path, "w")

        # all edges and utilizations
        edge_utl_path = f"{results_path}/{num_cluster}/edge_utls.txt"
        edge_utilizations = open(edge_utl_path, "w")


        responsibility_path = f"{results_path}/{num_cluster}/responsibility.txt"
        responsibility= open(responsibility_path, "w")
        # HACK END

    else:
        optimal_path = f"{results_path}/{num_cluster}/optimal_values_failure_id_{props.failure_id}.txt"
        optimal_values = open(optimal_path, "w")

    # HACK for getting the optimal ratios
    optimal_ratios = []
    for i, snapshot in tqdm.tqdm(enumerate(manifest[start_index:end_index]), total=len(manifest[start_index:end_index])):

        index = start_index + i
        topology_filename, pairs_filename, tm_filename = snapshot
        topology_filename = topology_filename.strip()
        pairs_filename = pairs_filename.strip()
        tm_filename = tm_filename.strip()

        previous_sp = copy.deepcopy(current_sp)
        current_sp = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename, dataset_dir)

        if (len(previous_sp.graph.nodes()) != len(current_sp.graph.nodes()))\
        or (set(previous_sp.graph.nodes()) != set(current_sp.graph.nodes()))\
        or (not np.array_equal(previous_sp.pairs, current_sp.pairs))\
        or(len(previous_sp.graph.edges()) != len(current_sp.graph.edges())):
            optimal_values.close()
            filenames.close()

            if num_snapshots_in_cluster > 0:
                num_cluster += 1
                num_snapshots_in_cluster = 0
            elif num_snapshots_in_cluster == 0:
                path1 = f"{dataset_dir}/topologies/paths"
                path2 = f"{dataset_dir}/topologies/paths_dict"
                os.remove(f"{path1}/{props.topo}_{props.num_paths_per_pair}_paths_cluster_{num_cluster}.pkl")
                os.remove(f"{path2}/{props.topo}_{props.num_paths_per_pair}_paths_dict_cluster_{num_cluster}.pkl")

            try:
                os.mkdir(f"{results_path}/{num_cluster}")
            except:
                pass

            optimal_path = f"{results_path}/{num_cluster}/optimal_values.txt"
            optimal_values = open(optimal_path, "w")
            filenames_path = f"{results_path}/{num_cluster}/filenames.txt"
            filenames = open(filenames_path, "w")
            # HACK for bottleneck edges
            btl_path = f"{results_path}/{num_cluster}/bottleneck_edges.txt"
            bottleneck_edges = open(btl_path, "w")
            # HACK END

            num_pairs = current_sp.num_demands
            cluster_info = Cluster_Info(current_sp, props, num_cluster, dataset_dir)
            cluster_info.pij = cluster_info.compute_ksp_paths(num_paths_per_pair, cluster_info.sp.pairs)
            cluster_info.paths_to_edges = csr_matrix(cluster_info.get_paths_to_edges_matrix(cluster_info.pij).to_dense().numpy())

        np_tm = current_sp.tm
        capacities = current_sp.capacities.numpy().astype(np.float64)

        #### Prepare the Gurobi model
        mlu = Model("MLU")
        mlu.setParam(GRB.Param.OutputFlag, 0)
        ## Done with Gurobi model

        vars_mlu = mlu.addMVar((current_sp.num_demands, num_paths_per_pair), lb=0.0,ub=GRB.INFINITY,
                        vtype=GRB.CONTINUOUS)

        mlu.addConstrs((vars_mlu[k, :].sum() == 1) for k in range(current_sp.num_demands))
        max_link_util = mlu.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="max_link_util")
        # # Done with Gurobi variables

        vars_mlu_tm = vars_mlu.reshape(-1, 1)*np_tm
        commodities_on_links = cluster_info.paths_to_edges.T @ vars_mlu_tm

        capacities_constraints = []
        for k, commodities_on_link_temp in enumerate(commodities_on_links):
            rhs = gp.LinExpr(max_link_util*capacities[k])
            constraint = mlu.addConstr(commodities_on_link_temp <= rhs)
            capacities_constraints.append(constraint)
        # Done with capacity constraints

        obj = gp.LinExpr(max_link_util)
        model_obj = mlu.setObjective(obj, GRB.MINIMIZE)
        mlu.optimize()
        edges_map = cluster_info.edges_map
        inverse_edges_map = {v: k for k, v in edges_map.items()}
        p2e = cluster_info.paths_to_edges  # (num_pairs(num_demands)*num_paths, num_edges)

        if mlu.status == GRB.status.OPTIMAL or mlu.status == GRB.OPTIMAL:
            if mlu.ObjVal < 0.01:
                continue
            else:
                # HACK for getting most congested linkk
                # Calculate actual utilization for each edge

                # commodities_on_links = cluster_info.paths_to_edges.T @ vars_mlu_tm
                edge_loads = commodities_on_links.getValue().flatten()
                edge_utils = np.zeros(len(capacities))

                edge_utils = edge_loads / capacities

                # HACK Find the most congested edges 
                eps = 1e-2  # Define a small epsilon value for comparison
                mlu_val = np.max(edge_utils)
                most_congested_idxs = np.where(np.abs(edge_utils - mlu_val) < eps)[0]
                # print('Edges map', edges_map)
                # print('Congested edge_loads', edge_loads[most_congested_idxs])
                # print('Congested edge_utils', edge_utils[most_congested_idxs])
                most_congested_edges = [inverse_edges_map[i] for i in most_congested_idxs]
                # print('Most congested edges')

                edge_util_dict = { 
                    f"{edge}": round(float(util), 4)
                    for edge, util in zip(edges_map, edge_utils)
                }
                json.dump(edge_util_dict, edge_utilizations)
                edge_utilizations.write("\n")
                # print(f"Most congested edge: {most_congested_edge}, Utilization: {most_congested_util:.3f}")
                # print(edge_util_dict)

                # print(f"bottleneck edge: {most_congested_edge}, utilization: {most_congested_util:.3f}")
                bottleneck_info = {
                    "bottleneck_edge": most_congested_edges,
                    "utilization": round(mlu_val, 3),
                    "eps": eps
                }
                bottleneck_msg = json.dumps(bottleneck_info)
                # bottleneck_msg = f"bottleneck edge: {most_congested_edges}, utilization: {mlu_val:.3f}, eps: {eps}"
                # print(bottleneck_msg)

                # breakpoint()
                # with open(f"{results_path}/{num_cluster}/bottleneck_edges.txt", 'w') as f:
                    # f.write(f"{topology_filename},{most_congested_edge},{most_congested_util:.6f}\n")
                bottleneck_edges.write(bottleneck_msg + "\n")
                # HACK END

                optimal_values.write(str(round(mlu.objVal, 9))+"\n")

                # HACK extract the optimal split ratios for building SFT dataset
                optimal_split_ratios = vars_mlu.X  # Extract the optimal variable values
                optimal_ratios.append(optimal_split_ratios)
                num_demands, num_paths = optimal_split_ratios.shape


                # # np_tm is repeated for k(num_tunnels) time
                # optimal_split_volumn = optimal_split_ratios.reshape(-1, 1)*np_tm
                # edge_volumn = p2e.T @ optimal_split_volumn
                # breakpoint()

                # np_tm current traffic matrix
                # p2e path to edges
                # num_paths, num_demands
                return_responsibility = False
                if return_responsibility == True:
                    all_responsibilities = {}
                    for edge_idx in most_congested_idxs:
                        edge = inverse_edges_map[edge_idx]
                        path_flows = optimal_split_ratios.reshape(-1, 1) * np_tm
                        edge_loads_matrix = p2e.T @ path_flows
                        edge_loads_matrix = edge_loads_matrix.flatten()
                        correct_edge_load = edge_loads_matrix[edge_idx]
                        
                        # responsibility traffic for congestions (Mbps)
                        responsibility_traffic = {}
                        for global_path_idx in range(p2e.shape[0]):

                            if p2e[global_path_idx, edge_idx] > 0:
                                dem_idx = global_path_idx // num_paths
                                path_idx = global_path_idx % num_paths
                                
                                # This is wrong: flow = np_tm[dem_idx].item() * optimal_split_ratios[dem_idx, path_idx]
                                flow = np_tm[global_path_idx].item() * optimal_split_ratios[dem_idx, path_idx]
                                if flow < 1e-8:
                                    continue
                                od_pair = tuple(current_sp.pairs[dem_idx])
                                
                                responsibility_traffic[f'{od_pair}'] = responsibility_traffic.get(f"{od_pair}", 0.0) + flow
                        all_responsibilities[f'{edge}'] = responsibility_traffic
                        total_responsibility = sum(responsibility_traffic.values())
                        if not  math.isclose(correct_edge_load, total_responsibility, rel_tol=1e-4):
                            print(correct_edge_load, total_responsibility)
                            breakpoint()
                    json.dump(all_responsibilities, responsibility)
                    responsibility.write("\n")


                # if mlu_val >1:
                #     breakpoint()

                # HACK END

                num_snapshots_in_cluster += 1
                if props.failure_id == None:
                    filenames.write(str(topology_filename) + "," + str(pairs_filename) + "," + str(tm_filename) + "\n")


    # HACK for getting the optimal ratios
    optimal_ratios_path = f"{results_path}/{num_cluster}/optimal_ratios.pkl"
    with open(optimal_ratios_path, 'wb') as file:
        pickle.dump(optimal_ratios, file)

    optimal_values.close()
    if props.failure_id == None:
        filenames.close()
    bottleneck_edges.close()
    edge_utilizations.close()
    responsibility.close()


if __name__ == "__main__":
    main()
