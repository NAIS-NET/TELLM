import os
import re
from typing import Optional
from openai import OpenAI
import torch
import numpy as np
from dataclasses import dataclass
from ..utils import logging, setup_rich_logging
setup_rich_logging()


@dataclass
class policyResponse:
    response: Optional[str] = None
    conversation_history: Optional[list[dict]] = None
    policy_embeddings: Optional[list[float]] = None


def chat(
        user_query, 
        system_prompt=None, 
        model="gpt-4.1",
        temperature=0.7,
        conversation_history=None,
        api_key=os.environ.get("OPENAI_API_KEY"), 
        base_url=os.environ.get("OPENAI_API_BASE_URL")
        ):
    """
    Uses OpenAI API to extract policy values from natural language query.
    
    Args:
        user_query (str): User's natural language request
        system_prompt (str): Optional system prompt to guide the model
        
    Returns:
        list: Three float values representing policy embeddings
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """
        You are TE-Head MCP, a specialized Model Context Protocol server that bridges intelligent network policy translation with traffic engineering optimization. Your role is to interpret natural language network objectives and translate them into precise three-dimensional policy embeddings for the TE controller.

        ## Core Responsibilities

        1. **Policy Embedding Translation**: Convert network objectives into [s_h, s_r, s_c] embeddings in [0,1]³
        2. **Context-Aware Reasoning**: Analyze network state, traffic patterns, and operational constraints
        3. **Feedback Integration**: Learn from optimization outcomes to refine translation strategies
        4. **Unified Interface**: Route commands between TE optimization and operational management

        ## Policy Embedding Dimensions

        ### History Dependency (s_h) $\in$ [0,1]
        - **0.0-0.5**: Reactive/instantaneous optimization, minimal historical consideration
        - **0.6-0.8**: Moderate history weighting, balanced responsiveness
        - **0.9-1.0**: Strong historical bias, robust burst headroom provisioning

        ### Global Robustness (s_r) $\in$ [0,1]
        - **0.0-0.3**: Best-effort routing, minimal redundancy
        - **0.4-0.7**: Balanced resilience with path diversity
        - **0.8-1.0**: Maximum robustness, ECMP-like worst-case provisioning

        ### Cost Sensitivity (s_c) $\in$ [0,1]
        - **0.0-0.1**: Performance-first, ignore costs
        - **0.2-0.5**: Balanced cost-performance trade-off
        - **0.6-1.0**: Cost-conscious, prioritize efficiency over peak performance

        ### Note
        - History Dependency tends to be more routine-oriented, while Global Robustness focuses on early warning for potential traffic bursts.
        - Cost Sensitivity encourages centralized path selection, while the other two dimensions promote diverse path selection. Avoid creating conflicting policies between these dimensions.

        ## Available Tools

        - `update_te_policy(embedding)`: Send new policy embedding to TE controller
        - `execute_operational_command(cmd)`: Route non-TE commands to network APIs
        - `log_feedback(embedding, metrics)`: Record policy performance for learning
        """
    
    # Initialize or use existing conversation history
    if conversation_history is None:
        conversation_history = [{"role": "system", "content": system_prompt}]
    
    # Add the new user message to conversation history
    conversation_history.append({"role": "user", "content": user_query})
    
    # Make API call with the full conversation history
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
        temperature=temperature, 
    )
    
    # Extract the response text
    response_text = response.choices[0].message.content

    conversation_history.append({"role": "assistant", "content": response_text})

    result = policyResponse(
        response=response_text,
        conversation_history=conversation_history,
        policy_embeddings=None
    )

    # Use regex to extract the policy values
    policy_match = re.search(r"# \[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", response_text)
    # policy_match = re.search(r"# \[([0-9.]+), ([0-9.]+), ([0-9.]+)\]", response_text)
    
    if policy_match:
        # Convert matched values to floats
        policy_values = [float(policy_match.group(1)), 
                         float(policy_match.group(2)), 
                         float(policy_match.group(3))]
        result.policy_embeddings = policy_values
    else:
        # Return default values if extraction fails
        logging.info("Warning: Failed to extract policy values, using defaults.")
        result.policy_embeddings = [0.0, 0.0, 0.0]

    return result

# Example usage:
# def run_inference_with_nlp_policy(query, dataset_idx=0):
#     """
#     Runs the TeLLM model with policys extracted from natural language.
    
#     Args:
#         query (str): Natural language query about network optimization preferences
#         dataset_idx (int): Index in the test dataset
        
#     Returns:
#         Model output with the specified policys
#     """
#     # Extract policys from natural language
#     policy_embeds = extract_policys_from_nlp(query)
#     print(f"Extracted policy embeddings: {policy_embeds}")
    
#     # Run the model with the extracted policys
#     result = pipe(test_dataset[dataset_idx], policy_embeds=policy_embeds, return_loss=True)
    
#     return result

# # Example query
# nlp_query = "I need a network configuration that prioritizes performance over reliability and cost."
# output = run_inference_with_nlp_policy(nlp_query, dataset_idx=0)

# # You can also create a batch processing function
# def batch_process_with_policy(query, start_idx=0, end_idx=10):
#     """Process multiple samples with the same policy"""
#     policy_embeds = extract_policys_from_nlp(query)
#     print(f"Using policy embeddings: {policy_embeds}")
    
#     all_results = []
#     for i in range(start_idx, end_idx):
#         result = pipe(test_dataset[i], policy_embeds=policy_embeds, return_loss=True)
#         all_results.append(result)
    
#     # Calculate average performance
#     avg_losses = {}
#     for key in all_results[0].loss.keys():
#         avg_losses[key] = sum(r.loss[key] for r in all_results) / len(all_results)
    
#     return avg_losses
