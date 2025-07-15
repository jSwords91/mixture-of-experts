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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        expert_outputs = torch.zeros_like(x, device=x.device)

        for i in range(self.num_experts):
            mask = dispatch_mask[:, i].bool()
            if mask.any():
                x_i = x[mask]
                y_i = self.experts[i](x_i)
                expert_outputs[mask] += y_i * scores[mask, i].unsqueeze(-1)

        logits = self.output_layer(expert_outputs)  # [B, num_classes]
        return logits, aux_loss, dispatch_mask, scores
