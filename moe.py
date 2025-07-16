import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_dim),
        )
    def forward(self, x):
        return self.net(x)

class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        assert self.k <= self.num_experts, "k must be <= num_experts"
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)  # [B, E]
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_indices = torch.topk(scores, self.k, dim=-1) # [B, k]

        dispatch_mask = torch.zeros_like(scores)
        dispatch_mask.scatter_(1, topk_indices, 1.0)

        load = dispatch_mask.sum(dim=0) # [E]
        importance = scores.sum(dim=0) # [E]
        B, E = scores.shape
        aux_loss = (importance * load).sum() * self.num_experts / (B ** 2)

        return dispatch_mask, scores, aux_loss, topk_indices

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_size, num_experts, top_k, num_classes):
        super().__init__()
        self.router = Router(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_size) for _ in range(num_experts)])
        self.output_layer = nn.Linear(input_dim, num_classes)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        dispatch_mask, scores, aux_loss, topk_indices = self.router(x)
        B, D = x.shape
        batch_times_top_k = B * self.top_k

        x_expanded = x.unsqueeze(1).expand(B, self.top_k, D)
        topk_scores = torch.gather(scores, 1, topk_indices)

        # Flatten inputs, scores, and expert indices for batch routing
        flat_inputs = x_expanded.reshape(batch_times_top_k, D)
        flat_scores = topk_scores.reshape(batch_times_top_k, 1)
        flat_expert_ids = topk_indices.reshape(batch_times_top_k)

        all_outputs = torch.zeros_like(flat_inputs)

        for i, expert in enumerate(self.experts):
            expert_mask = (flat_expert_ids == i)
            if expert_mask.any():
                x_i = flat_inputs[expert_mask]
                y_i = expert(x_i)
                all_outputs[expert_mask] = y_i

        all_outputs *= flat_scores
        expert_outputs = all_outputs.view(B, self.top_k, D).sum(dim=1)
        logits = self.output_layer(expert_outputs)
        return logits, aux_loss, dispatch_mask, scores

