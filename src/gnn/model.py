"""Edge-classification GNN: GINE layers with edge updates.

* Every account starts with the SAME learned vector (accounts have no
  attributes of their own in this dataset). Everything the model knows about an
  account comes from the transactions around it.
* GINE = Graph Isomorphism Network with Edge features. Each layer, an account
  sums up messages from its neighbors, where every message is
  relu(neighbor_vector + transaction_vector). Summing (not averaging) keeps
  count information, which matters: fan-in / fan-out is literally "how many".
* Edge updates: after each layer, each transaction's vector is refreshed using
  the two accounts at its ends (an idea from IBM's Multi-GNN work).
* To classify transaction u -> v we concatenate [account u, account v,
  transaction] and pass it through a small MLP.
"""
import torch
from torch import nn
from torch_geometric.nn import GINEConv


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class EdgeGINE(nn.Module):
    def __init__(self, edge_in, hidden=64, layers=2, dropout=0.1, edge_updates=True):
        super().__init__()
        self.node_init = nn.Parameter(torch.randn(hidden) * 0.1)
        self.edge_enc = mlp(edge_in + 1, hidden, hidden)  # +1 for the direction flag
        self.convs = nn.ModuleList(
            GINEConv(mlp(hidden, hidden, hidden), edge_dim=hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.edge_upd = nn.ModuleList(mlp(3 * hidden, hidden, hidden) for _ in range(layers)) \
            if edge_updates else None
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = dropout

    def forward(self, num_nodes, edge_index, msg_attr, tgt_src, tgt_dst, tgt_attr):
        x = self.node_init.expand(num_nodes, -1)
        e = self.edge_enc(msg_attr)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = torch.relu(norm(conv(x, edge_index, e)))
            x = x + nn.functional.dropout(h, self.dropout, self.training)
            if self.edge_upd is not None:
                s, d = edge_index
                e = e + self.edge_upd[i](torch.cat([x[s], x[d], e], dim=1))
        t = self.edge_enc(tgt_attr)
        return self.head(torch.cat([x[tgt_src], x[tgt_dst], t], dim=1)).squeeze(-1)
