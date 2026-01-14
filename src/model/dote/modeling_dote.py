import torch
from ..tellm.modeling_tellm import TeMetrics, TeOutput
from torch import nn


class DoteNetWork(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num):
        """Initialize the FigretNetWork with the network structure.

        Args:
            input_dim: dimension of input data, history len * flattened traffic matrix
            output_dim: dimension of output data, len of candidate paths all s-d pairs
            layer_num: number of hidden layers
        """
        super(DoteNetWork, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward the input data through the network.

        Args:
            x: input data, history len * flattened traffic matrix
        """
        x = self.flatten(x)
        logits = self.net(x)
        return logits




class DoteTeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hist_len = config.hist_len
        self.num_nodes = config.num_nodes
        self.num_paths = config.num_paths
        self.num_layers = config.num_layers

        self.input_dim = self.hist_len * self.num_nodes * (self.num_nodes - 1)
        self.output_dim = self.num_paths * self.num_nodes * (self.num_nodes - 1)
        self.model = DoteNetWork(self.input_dim, self.output_dim, self.num_layers).double()

    def forward(
            self, 
            te_inputs, 
            optimal_mlu=None,
            **kwars
            ):
        tms_hist = te_inputs['tms_hist'].double() # shape: (batch_size, hist_len, num_nodes * (num_nodes - 1))
        batch_size = tms_hist.shape[0]
        tms_hist = tms_hist[:, ::self.num_paths, :].contiguous()
        tms_hist.view(batch_size, -1) # shape: (batch_size, hist_len * num_nodes * (num_nodes - 1))


        # shape: (batch_size, num_paths)
        logits = self.model(tms_hist)

        loss, loss_val, max_sensitivity = self.loss(logits, te_inputs, optimal_mlu) if optimal_mlu is not None else (None, None)

        return TeOutput(
            gammas=logits,
            loss=TeMetrics(
                loss=loss,
                norm_mlu=loss_val,
                max_sensitivity=max_sensitivity
            )
        )

    def loss(self, y_pred_batch, te_inputs, optimal_mlu):
        """Compute the loss of the model.

        Args:
            y_pred: the split ratios for the candidate paths
            y_true: the true traffic demand and the optimal mlu
        """
        tm_raw = te_inputs['tms'].double()
        tm_hist = te_inputs['tms_hist'].double()
        capacities_raw = te_inputs['capacities'].double()
        paths_to_edges = te_inputs['paths_to_edges'].double()
        optimal_mlu = optimal_mlu.double()

        num_nodes = self.num_nodes
        losses = []
        loss_vals = []
        max_sensitivity_list = []
        batch_size = y_pred_batch.shape[0]
        for i in range(batch_size):
            y_pred = y_pred_batch[[i]]
            opt = optimal_mlu[[i]].item()
            y_true = tm_raw[[i]] #shape: (1, num_commodities)
            hist = tm_hist[[i]]
            capacities = capacities_raw[[i]]

            y_pred = y_pred + 1e-16

            y_pred = y_pred.view(1, -1, self.num_paths) #shape: (1, num_demands, num_paths_per_pair)
            y_pred_sum = y_pred.sum(dim=-1, keepdim=True)
            split_ratios = y_pred / y_pred_sum
            split_ratios = split_ratios.view(1, -1) #shape: (1, num_paths)
            edges_util_raw = self._compute_link_util(split_ratios, y_true, paths_to_edges, capacities)

            max_cong, _ = torch.max(edges_util_raw, dim=1)


            max_sensitivity = torch.max(split_ratios.view(-1, self.num_paths), dim = 1)[0] #shape: (num_commodities,)
            max_sensitivity_list.append(torch.mean(max_sensitivity))


            # loss function, the first term is the congestion, the second term is the sensitivity.
            # The operation of dividing by item() is used to balance different objectives, 
            # ensuring they are on the same scale. Then, alpha is used to adjust their importance.
            loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong / max_cong.item() 
            loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt
            losses.append(loss)
            loss_vals.append(loss_val)
        
        ret = sum(losses) / len(losses)
        ret_val = sum(loss_vals) / len(loss_vals)
        max_sensitivities = sum(max_sensitivity_list) / len(max_sensitivity_list)
        return ret, ret_val, max_sensitivities

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









