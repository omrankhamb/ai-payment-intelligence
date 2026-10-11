"""FraudGraph GNN Architectures: PureSAGE vs. HybridGatedGNN.

Implements:
1. FocalLoss (alpha=0.9, gamma=2.0) for 3.5% fraud class imbalance.
2. PureSAGE: Baseline GraphSAGE model (control).
3. HybridGatedConv: Parallel SAGEConv and GATv2Conv with learnable gated fusion:
   gate = sigmoid(W * x_dst / tau)
   out = gate * GAT(x) + (1 - gate) * SAGE(x)
4. HybridGatedGNN: 2-layer heterogeneous message-passing network with entity embeddings.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv
except ImportError:
    SAGEConv = None
    GATv2Conv = None
    HeteroConv = None


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance (3.5% fraud).

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Prevents gradient domination by overwhelming easy negative (legitimate) transactions.
    """

    def __init__(self, alpha: float = 0.90, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HybridGatedConv(nn.Module):
    """Hybrid convolution combining GraphSAGE (mean) and GATv2 (attention) via a learned gate.

    gate = sigmoid(W_gate * x_dst / tau)
    out = gate * GAT(x) + (1 - gate) * SAGE(x)
    """

    def __init__(
        self,
        in_channels: int | Tuple[int, int],
        out_channels: int,
        heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        if SAGEConv is not None:
            self.sage = SAGEConv(in_channels, out_channels)
            self.gat = GATv2Conv(
                in_channels,
                out_channels // heads,
                heads=heads,
                concat=True,
                add_self_loops=False,
            )
        else:
            self.sage = None
            self.gat = None

        dst_dim = in_channels[1] if isinstance(in_channels, tuple) else in_channels
        self.gate_proj = nn.Linear(dst_dim, out_channels)
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))
        self.last_gate_mean = 0.5

    def forward(
        self,
        x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.sage is None:
            raise RuntimeError("PyTorch Geometric is not installed.")

        out_sage = self.sage(x, edge_index)
        out_gat = self.gat(x, edge_index)

        dst_x = x[1] if isinstance(x, tuple) else x
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate = torch.sigmoid(self.gate_proj(dst_x) / temp)
        self.last_gate_mean = float(gate.mean().item())

        out = gate * out_gat + (1.0 - gate) * out_sage
        return out


class PureSAGEGNN(nn.Module):
    """Pure GraphSAGE heterogeneous baseline network (Control)."""

    def __init__(
        self,
        in_dim: int = 437,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_cards: int = 15000,
        num_merchants: int = 1000,
        num_devices: int = 3000,
    ):
        super().__init__()
        self.tx_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.card_emb = nn.Embedding(num_cards, hidden_dim)
        self.merch_emb = nn.Embedding(num_merchants, hidden_dim)
        self.dev_emb = nn.Embedding(num_devices, hidden_dim)

        # 2 Hetero SAGEConv layers
        self.conv1 = HeteroConv(
            {
                ("transaction", "uses_card", "card"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "at_merchant", "merchant"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "on_device", "device"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "velocity_cluster", "transaction"): SAGEConv(hidden_dim, hidden_dim),
                ("card", "rev_uses_card", "transaction"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("merchant", "rev_at_merchant", "transaction"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("device", "rev_on_device", "transaction"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "rev_velocity_cluster", "transaction"): SAGEConv(hidden_dim, hidden_dim),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("transaction", "uses_card", "card"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "at_merchant", "merchant"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "on_device", "device"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "velocity_cluster", "transaction"): SAGEConv(hidden_dim, out_dim),
                ("card", "rev_uses_card", "transaction"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("merchant", "rev_at_merchant", "transaction"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("device", "rev_on_device", "transaction"): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "rev_velocity_cluster", "transaction"): SAGEConv(hidden_dim, out_dim),
            },
            aggr="sum",
        )

        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        h_dict = {
            "transaction": self.tx_encoder(x_dict["transaction"]),
            "card": self.card_emb.weight if "card" not in x_dict else x_dict["card"],
            "merchant": self.merch_emb.weight if "merchant" not in x_dict else x_dict["merchant"],
            "device": self.dev_emb.weight if "device" not in x_dict else x_dict["device"],
        }

        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        logits = self.head(h_dict["transaction"]).squeeze(-1)
        return logits


class HybridGatedGNN(nn.Module):
    """Hybrid Gated GNN: SAGE (mean) + GATv2 (attention) per edge type."""

    def __init__(
        self,
        in_dim: int = 437,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_cards: int = 15000,
        num_merchants: int = 1000,
        num_devices: int = 3000,
    ):
        super().__init__()
        self.tx_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.card_emb = nn.Embedding(num_cards, hidden_dim)
        self.merch_emb = nn.Embedding(num_merchants, hidden_dim)
        self.dev_emb = nn.Embedding(num_devices, hidden_dim)

        # Layer 1 Hybrid Convs
        self.conv1 = HeteroConv(
            {
                ("transaction", "uses_card", "card"): HybridGatedConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "at_merchant", "merchant"): HybridGatedConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "on_device", "device"): HybridGatedConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "velocity_cluster", "transaction"): HybridGatedConv(hidden_dim, hidden_dim),
                ("card", "rev_uses_card", "transaction"): HybridGatedConv((hidden_dim, hidden_dim), hidden_dim),
                ("merchant", "rev_at_merchant", "transaction"): HybridGatedConv((hidden_dim, hidden_dim), hidden_dim),
                ("device", "rev_on_device", "transaction"): HybridGatedConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "rev_velocity_cluster", "transaction"): HybridGatedConv(hidden_dim, hidden_dim),
            },
            aggr="sum",
        )

        # Layer 2 Hybrid Convs
        self.conv2 = HeteroConv(
            {
                ("transaction", "uses_card", "card"): HybridGatedConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "at_merchant", "merchant"): HybridGatedConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "on_device", "device"): HybridGatedConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "velocity_cluster", "transaction"): HybridGatedConv(hidden_dim, out_dim),
                ("card", "rev_uses_card", "transaction"): HybridGatedConv((hidden_dim, hidden_dim), out_dim),
                ("merchant", "rev_at_merchant", "transaction"): HybridGatedConv((hidden_dim, hidden_dim), out_dim),
                ("device", "rev_on_device", "transaction"): HybridGatedConv((hidden_dim, hidden_dim), out_dim),
                ("transaction", "rev_velocity_cluster", "transaction"): HybridGatedConv(hidden_dim, out_dim),
            },
            aggr="sum",
        )

        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        h_dict = {
            "transaction": self.tx_encoder(x_dict["transaction"]),
            "card": self.card_emb.weight if "card" not in x_dict else x_dict["card"],
            "merchant": self.merch_emb.weight if "merchant" not in x_dict else x_dict["merchant"],
            "device": self.dev_emb.weight if "device" not in x_dict else x_dict["device"],
        }

        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        logits = self.head(h_dict["transaction"]).squeeze(-1)
        return logits

    def get_gate_statistics(self) -> Dict[str, float]:
        """Collect gate averages across edge types (Attention vs Mean weight)."""
        stats = {}
        for edge_type, module in self.conv1.convs.items():
            if isinstance(module, HybridGatedConv):
                key = f"{edge_type[1]}_L1"
                stats[key] = round(module.last_gate_mean, 4)
        for edge_type, module in self.conv2.convs.items():
            if isinstance(module, HybridGatedConv):
                key = f"{edge_type[1]}_L2"
                stats[key] = round(module.last_gate_mean, 4)
        return stats
