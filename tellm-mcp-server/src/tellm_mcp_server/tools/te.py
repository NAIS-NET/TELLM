from mcp.server.fastmcp import FastMCP
from tellm_mcp_server.api_client import make_onos_request, make_tellm_request
from typing import Annotated
from collections import deque

history_policy_embeddings = deque(maxlen=12)

async def tellm_push_policy_embeddings(
    history_score: Annotated[float, "History dependency score (s_h) in [0, 1]: Trade-off between MLU optimization and demand-specific resilience. Lower values (0.0) assume reliable traffic patterns, minimizing MLU for predictable demands, while higher values (1.0) may increase MLU by allocating more capacity to historically volatile traffic demands."],
    global_score: Annotated[float, "Global robustness score (s_r) in [0, 1]: Trade-off between MLU efficiency and network-wide resilience. Lower values (0.0) minimize current MLU with best-effort routing, while higher values (1.0) may increase MLU by uniformly reserving capacity across all demand pairs regardless of their historical volatility."],
    cost_score: Annotated[float, "Cost sensitivity score (s_c) in [0, 1]: Trade-off between MLU minimization and operational costs. Lower values (0.0) prioritize minimal MLU regardless of resource costs, while higher values (1.0) accept higher MLU to reduce network infrastructure expenses."]
) -> str:
    """Push the current policy embeddings to the TE-head of TeLLM for different traffic engineering preferences. Note: embedding [0, 0, 0] already will optimize the main objective (MLU), which will gain the optimal solution under static demands without cost concerns, i.e., when traffics are stable/consistent; each unnecessary score assignment would lead to traffic engineering degradation."""
    
    # Store in history
    history_policy_embeddings.append((history_score, global_score, cost_score))
    
    # Prepare the policy embeddings for the API call
    policy_embeds = [history_score, global_score, cost_score]
    
    try:
        # Use the new make_tellm_request function
        response = await make_tellm_request(
            "POST",
            "/api/policy",
            json={"policy_embeds": policy_embeds}
        )
        
        return f"Policy embeddings successfully pushed: [s_h={history_score}, s_r={global_score}, s_c={cost_score}]. Response: {response}. History: {history_policy_embeddings}"
                
    except Exception as e:
        return f"Error pushing policy embeddings: {str(e)}"

async def get_tellm_topology_statistics() -> str:
    """
    Get comprehensive statistical analysis of the network topology from TeLLM with rounded values.
    Provides detailed metrics about network structure, connectivity, and capacity.
    """
    try:
        # Make request to topology stats API
        stats_data = await make_tellm_request("GET", "/api/topology/stats")
        
        # Format the response into a comprehensive summary
        summary = ["# TeLLM Network Topology Statistics\n"]
        
        # Basic statistics
        basic = stats_data.get('basic_stats', {})
        summary.append(f"## Basic Network Metrics")
        summary.append(f"- Nodes: {basic.get('node_count', 0)}")
        summary.append(f"- Edges: {basic.get('edge_count', 0)}")
        summary.append(f"- Density: {basic.get('density', 0)}")
        summary.append(f"- Directed: {'Yes' if basic.get('is_directed') else 'No'}")
        
        # Degree statistics
        degree_stats = stats_data.get('degree_stats', {})
        summary.append(f"\n## Degree Distribution")
        if basic.get('is_directed'):
            summary.append(f"### In-Degree")
            in_deg = degree_stats.get('in_degree', {})
            summary.append(f"  - Min: {in_deg.get('min', 0)}")
            summary.append(f"  - Max: {in_deg.get('max', 0)}")
            summary.append(f"  - Mean: {in_deg.get('mean', 0)}")
            summary.append(f"  - Median: {in_deg.get('median', 0)}")
            
            summary.append(f"### Out-Degree")
            out_deg = degree_stats.get('out_degree', {})
            summary.append(f"  - Min: {out_deg.get('min', 0)}")
            summary.append(f"  - Max: {out_deg.get('max', 0)}")
            summary.append(f"  - Mean: {out_deg.get('mean', 0)}")
            summary.append(f"  - Median: {out_deg.get('median', 0)}")
        else:
            summary.append(f"  - Min: {degree_stats.get('min', 0)}")
            summary.append(f"  - Max: {degree_stats.get('max', 0)}")
            summary.append(f"  - Mean: {degree_stats.get('mean', 0)}")
            summary.append(f"  - Median: {degree_stats.get('median', 0)}")
        
        # Component statistics
        comp_stats = stats_data.get('component_stats', {})
        summary.append(f"\n## Connectivity Analysis")
        if basic.get('is_directed'):
            summary.append(f"- Strongly connected: {comp_stats.get('is_strongly_connected', False)}")
            summary.append(f"- Weakly connected: {comp_stats.get('is_weakly_connected', False)}")
            summary.append(f"- Strongly connected components: {comp_stats.get('num_strongly_connected', 0)}")
            summary.append(f"- Weakly connected components: {comp_stats.get('num_weakly_connected', 0)}")
            summary.append(f"- Largest strongly connected component: {comp_stats.get('largest_strongly_component_size', 0)} nodes")
            summary.append(f"- Largest weakly connected component: {comp_stats.get('largest_weakly_component_size', 0)} nodes")
        else:
            summary.append(f"- Connected: {comp_stats.get('is_connected', False)}")
            summary.append(f"- Connected components: {comp_stats.get('num_connected_components', 0)}")
            summary.append(f"- Largest component: {comp_stats.get('largest_component_size', 0)} nodes")
            if comp_stats.get('diameter'):
                summary.append(f"- Diameter: {comp_stats.get('diameter')}")
            if comp_stats.get('avg_shortest_path'):
                summary.append(f"- Average shortest path: {comp_stats.get('avg_shortest_path')}")
        
        # Clustering coefficient
        cluster_stats = stats_data.get('clustering_stats', {})
        if 'avg_clustering_coefficient' in cluster_stats:
            summary.append(f"\n## Clustering")
            summary.append(f"- Average clustering coefficient: {cluster_stats.get('avg_clustering_coefficient', 0)}")
        
        # Capacity statistics
        capacity_stats = stats_data.get('capacity_stats', {})
        if capacity_stats and not any('error' in k for k in capacity_stats.keys()):
            summary.append(f"\n## Capacity Distribution")
            summary.append(f"- Minimum capacity: {capacity_stats.get('min', 0)}")
            summary.append(f"- Maximum capacity: {capacity_stats.get('max', 0)}")
            summary.append(f"- Average capacity: {capacity_stats.get('mean', 0)}")
            summary.append(f"- Total capacity: {capacity_stats.get('total', 0)}")
            summary.append(f"- Median capacity: {capacity_stats.get('median', 0)}")
            summary.append(f"- Units: {capacity_stats.get('units', 'N/A')}")
        
        # Centrality statistics
        centrality_stats = stats_data.get('centrality_stats', {})
        if centrality_stats and 'betweenness' in centrality_stats:
            betw = centrality_stats['betweenness']
            summary.append(f"\n## Centrality (Betweenness)")
            summary.append(f"- Maximum value: {betw.get('max_value', 0)}")
            summary.append(f"- Minimum value: {betw.get('min_value', 0)}")
            summary.append(f"- Average value: {betw.get('mean', 0)}")
            summary.append(f"- Sample size: {betw.get('sample_size', 0)} nodes")
        
        return "\n".join(summary)
        
    except Exception as e:
        return f"Error retrieving TeLLM topology statistics: {str(e)}"

async def get_tellm_performance_stats(
    wait: Annotated[float, "Wait time in seconds before collecting results. Use 0 for immediate response or positive values to wait for more data accumulation. Default is 0. Recommended to use wait time only when monitoring performance after policy embedding updates, typically 5-12 seconds."] = 0
) -> str:
    """
    Get current network performance statistics from TeLLM including MLU, sensitivity, and policy embeddings.
    Provides insights into traffic engineering optimization results.
    Optionally wait for specified time to collect more data samples.
    """
    try:
        # Prepare query parameters
        params = {}
        if wait > 0:
            params['wait'] = wait
        
        # Make request to performance stats API with optional wait parameter
        stats_data = await make_tellm_request("GET", "/api/stats", params=params)
        
        if 'error' in stats_data:
            return f"No TeLLM performance data available. Current policy embeddings: {stats_data.get('policy_embeddings', [])}"
        
        summary = ["# TeLLM Network Performance Statistics\n"]
        
        # Add wait time information if applied
        if wait > 0:
            summary.append(f"*Data collected after waiting {wait:.1f} seconds*\n")
        
        # MLU statistics
        mlu_stats = stats_data.get('mlu', {})
        summary.append(f"## Maximum Link Utilization (MLU)")
        summary.append(f"- Minimum: {mlu_stats.get('min', 0):.4f}")
        summary.append(f"- Maximum: {mlu_stats.get('max', 0):.4f}")
        summary.append(f"- Average: {mlu_stats.get('mean', 0):.4f}")
        summary.append(f"- Median: {mlu_stats.get('median', 0):.4f}")
        summary.append(f"- Standard deviation: {mlu_stats.get('std', 0):.4f}")
        
        # Max sensitivity statistics
        sens_stats = stats_data.get('max_sensitivity', {})
        summary.append(f"\n## Maximum Sensitivity")
        summary.append(f"- Minimum: {sens_stats.get('min', 0):.4f}")
        summary.append(f"- Maximum: {sens_stats.get('max', 0):.4f}")
        summary.append(f"- Average: {sens_stats.get('mean', 0):.4f}")
        summary.append(f"- Median: {sens_stats.get('median', 0):.4f}")
        summary.append(f"- Standard deviation: {sens_stats.get('std', 0):.4f}")
        
        # Cost statistics (if available)
        cost_stats = stats_data.get('cost', {})
        if cost_stats:
            summary.append(f"\n## Cost Statistics")
            summary.append(f"- Minimum: {cost_stats.get('min', 0):.4f}")
            summary.append(f"- Maximum: {cost_stats.get('max', 0):.4f}")
            summary.append(f"- Average: {cost_stats.get('mean', 0):.4f}")
            summary.append(f"- Median: {cost_stats.get('median', 0):.4f}")
            summary.append(f"- Standard deviation: {cost_stats.get('std', 0):.4f}")
        
        # Policy embeddings
        policy_embeds = stats_data.get('policy_embeddings', [])
        summary.append(f"\n## Current Policy Embeddings")
        if policy_embeds:
            summary.append(f"- History dependency (s_h): {policy_embeds[0]:.3f}")
            summary.append(f"- Global robustness (s_r): {policy_embeds[1]:.3f}")
            summary.append(f"- Cost sensitivity (s_c): {policy_embeds[2]:.3f}")
        else:
            summary.append("- No active policy embeddings")
        
        # Samples processed
        samples = stats_data.get('samples_processed', 0)
        summary.append(f"\n## Data Processing")
        summary.append(f"- Samples processed: {samples}")
        
        # Applied wait time
        applied_wait = stats_data.get('wait_time_applied', 0)
        if applied_wait > 0:
            summary.append(f"- Applied wait time: {applied_wait:.1f} seconds")
        
        return "\n".join(summary)
        
    except Exception as e:
        return f"Error retrieving TeLLM performance statistics: {str(e)}"

def register_tools(mcp_server: FastMCP):
    mcp_server.tool()(tellm_push_policy_embeddings)
    mcp_server.tool()(get_tellm_topology_statistics)
    mcp_server.tool()(get_tellm_performance_stats)
