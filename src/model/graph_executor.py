import torch
import torch.nn as nn
import random 
import networkx as nx 
from torch_geometric.utils import degree

from src.model.processors import BipartiteMPNN


class GraphNeuralExecutor(nn.Module):
    def __init__(self, hidden_dim, eps):
        super(GraphNeuralExecutor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.eps = eps

        # Encoder  
        self.x_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.degree_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.ReLU(),
        )

        # Processor
        self.processor = BipartiteMPNN(
            hidden_dim=hidden_dim
        )

        # Decoder
        self.x_del_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True)
            # No activation function due to BCEWithLogitsLoss
        )
        self.x_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.ReLU()
        )
        self.y_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.ReLU()
        )
        self.eps_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.ReLU()
        )

    def mse_loss(self, pred, true, mask=None):
        loss_fn = nn.MSELoss(reduction='mean')
        if mask is not None and mask.sum() == 0:
            return None
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        loss = loss_fn(pred, true).squeeze(-1)
        assert pred.shape == true.shape
        return loss

    def bce_loss(self, pred, true, mask=None):
        if mask is not None and mask.sum() == 0:
            return None
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        assert pred.shape == true.shape
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn(pred, true).squeeze(-1)
        return loss

    def accuracy(self, pred, true, mask=None):
        if mask is not None and mask.sum() == 0:
            return None
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
        assert pred.shape == true.shape
        acc = torch.sum(pred == true) / len(pred)
        return acc

    def teacher_forcing(self, batch, current_epoch=None):
        # Dimension information
        #   x: (batch_size * num_elements, timesteps)
        #   y: (batch_size * num_sets, timesteps)
        num_x = batch.x.shape[0]
        num_y = batch.y.shape[0]
        timesteps = batch.x.shape[1]
        x_weights = batch.x[:, 0]
        edge_index = batch.edge_index
        batch_index_x = batch.x_batch
        batch_index_y = batch.y_batch
        batch_size = torch.max(batch_index_x).item() + 1
        
        # Initialization
        all_x_del_loss = []
        all_x_loss = [] 
        all_y_loss = []
        if self.eps:
            all_eps_loss = []
        pred_set = torch.zeros(num_x).unsqueeze(-1).to(batch.x.device)
        pred_set_prob = torch.zeros(num_x).unsqueeze(-1).to(batch.x.device)
        pred_y_mask = torch.tensor([True for _ in range(num_y)]).to(batch.x.device)

        # Encoders  
        x_pred = batch.x[:, 0]
        prev_x = self.x_encoder(batch.x[:, :-1])   # teacher forcing hints

        # Compute next step values
        for t in range(1, timesteps):
            # Get all the true values
            x_del_true = batch.x_mask[:, t] - batch.x_mask[:, t - 1]  # Which nodes are deleted at timestep t
            x_true = batch.x[:, t]
            y_true = batch.y[:, t]
            x_mask_true = ~ batch.x_mask[:, t - 1].bool() 
            y_mask_true = batch.y_mask[:, t - 1]
            if self.eps:
                eps_true = batch.eps[:, t]
                eps_mask_true = batch.eps_mask[:, t - 1]
            
            # Get edge mask
            assert x_mask_true.shape[0] > torch.max(edge_index[1]).item()
            assert y_mask_true.shape[0] > torch.max(edge_index[0]).item()
            edge_mask = (x_mask_true[edge_index[1]] & y_mask_true[edge_index[0]]).squeeze(-1)
            if edge_mask.sum() == 0:
                break

            # Compute degree features
            x_degree = degree(edge_index[:, edge_mask][1], num_x).unsqueeze(-1)
            x_degree = torch.log(x_degree + 1)
            h_x_degree = self.degree_encoder(x_degree)

            # Processor (with noisy teacher forcing) 
            h_x = self.x_encoder(x_pred)
            if self.training: 
                tf_prob = torch.rand(h_x.shape[0]).unsqueeze(-1).to(h_x.device)
                tf_prob = tf_prob.expand(-1, h_x.shape[1])
                h_x = torch.where(tf_prob > 0.5, prev_x[:, t - 1], h_x)
            if self.eps:
                next_x, next_y, next_eps = self.processor(h_x=h_x, h_x_degree=h_x_degree, edge_index=edge_index, x_mask=x_mask_true, y_mask=y_mask_true, 
                                                          edge_mask=edge_mask, batch_index_x=batch_index_x, batch_index_y=batch_index_y, batch_size=batch_size, eps=True)
            else:
                next_x, next_y, _ = self.processor(h_x=h_x, h_x_degree=h_x_degree, edge_index=edge_index, x_mask=x_mask_true, y_mask=y_mask_true, 
                                                   edge_mask=edge_mask, batch_index_x=batch_index_x, batch_index_y=batch_index_y, batch_size=batch_size, eps=False)

            # Decoders
            x_del_pred = self.x_del_decoder(next_x)
            x_pred = self.x_decoder(next_x)
            y_pred = self.y_decoder(next_y)
            if self.eps:
                eps_pred = self.eps_decoder(next_eps)

            # Compute losses and accuracies
            x_del_loss = self.bce_loss(x_del_pred, x_del_true, mask=x_mask_true) 
            x_loss = self.mse_loss(x_pred, x_true, x_mask_true)
            y_loss = self.mse_loss(y_pred, y_true, y_mask_true)
            if self.eps:
                eps_loss  = self.mse_loss(eps_pred, eps_true, eps_mask_true)
            if x_del_loss is not None:
                all_x_del_loss += [x_del_loss]
            if x_loss is not None:
                all_x_loss += [x_loss]
            if y_loss is not None:
                all_y_loss += [y_loss]
            if self.eps and eps_loss is not None:
                all_eps_loss += [eps_loss] 

            # Update pred_set and adjacency matrix
            x_del_pred_sig = torch.sigmoid(x_del_pred)
            pred_set[(x_del_pred_sig > 0.5) & (pred_set == 0)] = 1
            pred_set_prob = pred_set_prob * (1 - x_mask_true.float()) + x_del_pred * x_mask_true.float()
            
            # Terminate early if batched graph has no LHS nodes (i.e. all sets covered)
            if not torch.any(y_mask_true):
                break

        # Calculate optimal loss and accuracy
        optimal_set_loss = self.bce_loss(pred_set_prob.squeeze(-1), batch.primal_optimal_solution.float())
        optimal_set_acc = self.accuracy(pred_set.squeeze(-1), batch.primal_optimal_solution)

        # Aggregate losses
        x_del_loss = torch.stack(all_x_del_loss, dim=-1).mean()
        x_loss = torch.stack(all_x_loss, dim=-1).mean()
        y_loss = torch.stack(all_y_loss, dim=-1).mean()
        if self.eps:
            eps_loss = torch.stack(all_eps_loss, dim=-1).mean()

        # Aggregate accuracies
        final_set = batch.x_mask[:, -1]
        final_set_acc = self.accuracy(pred_set, final_set)

        # Aggregate ratios
        assert x_weights.shape == pred_set.shape 
        assert x_weights.shape == final_set.shape
        weight = torch.sum(x_weights * pred_set).float()
        final_weight_ratio = weight / torch.sum(x_weights * final_set)
        optimal_weight_ratio =  weight / batch.primal_optimal_weight.sum()

        results = {
            "x_del_loss": x_del_loss,
            "x_loss": x_loss,
            "y_loss": y_loss,
            "optimal_set_loss": optimal_set_loss, 
            "optimal_set_acc": optimal_set_acc,
            "final_set_acc": final_set_acc,
            "final_weight_ratio": final_weight_ratio, 
            "optimal_weight_ratio": optimal_weight_ratio
        }
        if self.eps:
            results["eps_loss"] = eps_loss
        return results 

    def test(self, batch):
        # Get dimensions
        num_x = batch.x.shape[0]
        num_y = batch.y.shape[0]
        timesteps = batch.x.shape[1]
        edge_index = batch.edge_index
        x_weights = batch.x[:, 0]
        batch_index_x = batch.x_batch
        batch_index_y = batch.y_batch
        batch_size = torch.max(batch_index_x).item() + 1

        # Initialization
        x_pred = batch.x[:, 0]
        pred_set = torch.zeros(num_x).unsqueeze(-1).to(batch.x.device)
        pred_y_mask = torch.tensor([True for _ in range(num_y)]).unsqueeze(-1).to(batch.x.device)
        graph_x_finishes = torch.zeros((num_x, ), device=batch.x.device).bool()

        # Algorithm simulation 
        for _ in range(timesteps):  
            # Encoder 
            h_x = self.x_encoder(x_pred)

            # Get edge mask
            pred_x_mask = (1 - pred_set).bool()
            edge_mask = (pred_x_mask[edge_index[1]] & pred_y_mask[edge_index[0]]).squeeze(-1)
            if edge_mask.sum() == 0:
                break

            # Compute degree features
            x_degree = degree(edge_index[:, edge_mask][1], num_x).unsqueeze(-1)
            x_degree = torch.log(x_degree + 1)  
            h_x_degree = self.degree_encoder(x_degree)

            # Processor 
            if self.eps:
                next_x, _, _ = self.processor(h_x=h_x, h_x_degree=h_x_degree, edge_index=edge_index, x_mask=pred_x_mask, y_mask=pred_y_mask, 
                                              edge_mask=edge_mask, batch_index_x=batch_index_x, batch_index_y=batch_index_y, batch_size=batch_size, eps=True)
            else:
                next_x, _, _ = self.processor(h_x=h_x, h_x_degree=h_x_degree, edge_index=edge_index, x_mask=pred_x_mask, y_mask=pred_y_mask, 
                                              edge_mask=edge_mask, batch_index_x=batch_index_x, batch_index_y=batch_index_y, batch_size=batch_size, eps=False)

            # Decoder 
            x_del_pred = self.x_del_decoder(next_x)
            x_del_pred = torch.sigmoid(x_del_pred)
            x_pred = self.x_decoder(next_x)

            # Get the x nodes to be deleted
            if self.eps:
                x_del_pred[pred_set.squeeze(-1) == 1] = 0.
                max_probs = torch.zeros((batch_size, ), device=x_del_pred.device)
                max_probs = max_probs.scatter_reduce_(dim=0, index=batch_index_x, src=x_del_pred.squeeze(-1), reduce='amax', include_self=False)
                delete_x_mask = (torch.abs(x_del_pred.squeeze(-1) - max_probs[batch_index_x]) < 1e-8)
            else:
                x_del_pred[pred_set.squeeze(-1) == 1] = 0.
                delete_x_mask = (x_del_pred > 0.5).squeeze(-1)
            delete_x_mask = delete_x_mask & (~pred_set.squeeze(-1).bool()) & (~graph_x_finishes)
            pred_set[delete_x_mask] = 1

            # Update pred_y_mask by removing all y nodes connected to deleted x nodes
            delete_x_indices = torch.nonzero(delete_x_mask).squeeze(-1)
            deleted_edges_mask = torch.isin(edge_index[1], delete_x_indices)
            delete_y_indices = edge_index[0][deleted_edges_mask].unique()
            pred_y_mask[delete_y_indices] = False
            
            # Mask out a graph in the batch if all y nodes are removed
            graph_finishes = torch.zeros((batch_size, ), device=x_del_pred.device)
            graph_finishes = graph_finishes.scatter_reduce_(dim=0, index=batch_index_y, src=pred_y_mask.squeeze(-1).float(), reduce='amax', include_self=False)
            graph_finishes = ~(graph_finishes.bool())
            graph_x_finishes = graph_finishes[batch_index_x]

            # Terminate if the batched graph is empty
            if not torch.any(pred_y_mask):
                break

        # Corrective stage with greedy algorithm if the output is not valid 
        if torch.any(pred_y_mask):
            while torch.any(pred_y_mask):
                pred_x_mask = (1 - pred_set).bool()
                edge_mask = (pred_x_mask[edge_index[1]] & pred_y_mask[edge_index[0]]).squeeze(-1)
                x_degree = degree(edge_index[:, edge_mask][1], num_x).unsqueeze(-1)
                x_indices = torch.tensor([i for i in range(num_x)]).unsqueeze(-1).to(batch.x.device)
                x_indices = x_indices[pred_x_mask & ~graph_x_finishes.unsqueeze(-1)]
                x_remain_weights = x_weights[pred_x_mask & ~graph_x_finishes.unsqueeze(-1)] 
                x_degree = x_degree[pred_x_mask & ~graph_x_finishes.unsqueeze(-1)]
                x_remain_weights_degree = x_remain_weights / x_degree 
                min_x_remain_weights_degree = torch.argmin(x_remain_weights_degree)
                min_x = x_indices[min_x_remain_weights_degree]
                # Delete min_x
                delete_x_indices = torch.tensor([min_x]).to(batch.x.device)
                pred_set[min_x] = 1
                deleted_edges_mask = torch.isin(edge_index[1], delete_x_indices)
                delete_y_indices = edge_index[0][deleted_edges_mask].unique()
                pred_y_mask[delete_y_indices] = False

        # Caculate accuracies compared to final and optimal sets 
        final_set = batch.x_mask[:, -1]
        final_set_acc = self.accuracy(pred_set, final_set)
        assert x_weights.shape == pred_set.shape
        assert x_weights.shape == final_set.shape
        weight = torch.sum(x_weights * pred_set).float()
        final_weight_ratio = weight / torch.sum(x_weights * final_set)
        optimal_weight_ratio = weight / batch.primal_optimal_weight.sum()
        optimal_set_acc = self.accuracy(pred_set.squeeze(-1), batch.primal_optimal_solution)

        results = {
            "final_set_acc": final_set_acc,
            "final_weight_ratio": final_weight_ratio,
            "optimal_weight_ratio": optimal_weight_ratio,
            "optimal_set_acc": optimal_set_acc,
            "pred_set": pred_set.detach().clone(),
        }

        return results 
    