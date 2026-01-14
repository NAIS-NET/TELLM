import torch
from transformers import AutoModel


class TeLLMInferencePipline:
    def __init__(self, pretrained_path=None, model=None, device='cuda:0', torch_dtype=torch.float32) -> None:
        assert pretrained_path is not None or model is not None, "Either pretrained_path or model must be provided."
        assert pretrained_path is None or model is None, "Only one of pretrained_path or model should be provided."
        self.pretrained_path = pretrained_path
        self.device = device
        self.torch_dtype = torch_dtype

        if model is not None:
            self.model = model.to(device=device)
        else:
            self.model = AutoModel.from_pretrained(
                self.pretrained_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        self.model= self.model.eval().to(device=device)
    def __call__(self, te_sample, policy_embeds=None, optimal_mlu=None, return_loss=False, num_for_loops=6,num_paths_per_pair=4):
        te_sample, opt = self.preprocess(te_sample, device=self.device)
        if policy_embeds is None:
            policy_embeds = [0.0, 0.0, 0.0]
        policy_embeds = torch.as_tensor(policy_embeds, dtype=torch.float32).unsqueeze(0).to(device=self.device)
        if policy_embeds.any() > 1.0:
            raise ValueError("Policy embeddings should be in the range [0, 1]")
        with torch.no_grad():
            outputs = self.model(
                te_sample, 
                policy_embeds=policy_embeds, 
                optimal_mlu=opt,
                num_for_loops=num_for_loops,
                num_paths_per_pair=num_paths_per_pair, 
                return_loss=return_loss)
        return outputs

    def preprocess(self, item, device=None):
        """
        Preprocess a single dataset item to te_inputs format.
        
        Args:
            item: Dictionary containing raw data for a single sample
            device: Device to move tensors to (optional)
            
        Returns:
            Dictionary with te_inputs format suitable for model inference
        """
        # Convert tensors and add batch dimension
        tms_hist = torch.as_tensor(item['tms_hist'], dtype=torch.float32).unsqueeze(0)
        opt = torch.as_tensor(item['optimal_values'], dtype=torch.float32).unsqueeze(0)
        tms = torch.as_tensor(item['tms'], dtype=torch.float32).unsqueeze(0)
        tms_pred = torch.as_tensor(item['tms_pred'], dtype=torch.float32).unsqueeze(0)
        capacities = torch.as_tensor(item['capacities'], dtype=torch.float32).unsqueeze(0)
        node_features = torch.as_tensor(item['node_features'], dtype=torch.float32).unsqueeze(0)
        node_features_norm = torch.as_tensor(item['node_features_norm'], dtype=torch.float32).unsqueeze(0)
        
        # Move to device if specified
        if device is not None:
            tms_hist = tms_hist.to(device)
            opt = opt.to(device)
            tms = tms.to(device)
            tms_pred = tms_pred.to(device)
            capacities = capacities.to(device)
            node_features = node_features.to(device)
            node_features_norm = node_features_norm.to(device)
        
        # Graph structure (may need device transfer too depending on your setup)
        edge_index = item['edge_index'].to(device)
        pte = item['paths_to_edges'].to(device)
        padded_edge_ids_per_path = item['padded_edge_ids_per_path'].to(device)
        
        te_inputs = {
            'node_features': node_features,
            'node_features_norm': node_features_norm,
            'capacities': capacities,
            'tms': tms,
            'tms_pred': tms_pred,
            'tms_hist': tms_hist,
            'edge_index': edge_index,
            'paths_to_edges': pte,
            'padded_edge_ids_per_path': padded_edge_ids_per_path,
        }
        
        return te_inputs, opt