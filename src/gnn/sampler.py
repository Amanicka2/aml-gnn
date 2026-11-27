"""A small neighbor sampler, written by hand.

Why: the full graph (5M edges) is far too big to push through a GNN at once on
a laptop. Instead, for each mini-batch of transactions we want to classify, we
grab a small random neighborhood around their two accounts (like PyG's
NeighborLoader does, but without needing its compiled extension library).

Direction matters for laundering (fan-in vs fan-out), so every transaction
u -> v becomes TWO message directions:
  * forward:  u sends a message to v   (rev = 0)
  * reverse:  v sends a message to u   (rev = 1)
The rev flag is appended to the edge features so the model can tell them apart.
"""
import torch


class IncidenceCSR:
    """For every account, the list of (neighbor, transaction id, direction) it touches."""

    def __init__(self, src, dst, edge_ids, num_nodes):
        center = torch.cat([dst[edge_ids], src[edge_ids]])
        nbr = torch.cat([src[edge_ids], dst[edge_ids]])
        eid = torch.cat([edge_ids, edge_ids])
        rev = torch.cat([torch.zeros(len(edge_ids), dtype=torch.int8),
                         torch.ones(len(edge_ids), dtype=torch.int8)])
        order = torch.argsort(center, stable=True)
        self.nbr, self.eid, self.rev = nbr[order], eid[order], rev[order]
        counts = torch.bincount(center, minlength=num_nodes)
        self.ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        self.ptr[1:] = torch.cumsum(counts, 0)
        del center, order

    def sample(self, nodes, k, gen=None):
        """Up to ~k random incident transactions per node (sampled with replacement, then de-duplicated)."""
        start = self.ptr[nodes]
        deg = self.ptr[nodes + 1] - start
        keep = deg > 0
        start, deg = start[keep], deg[keep]
        if len(start) == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, empty, torch.empty(0, dtype=torch.int8)
        r = (torch.rand(len(start), k, generator=gen) * deg[:, None]).long()
        pos = torch.unique((start[:, None] + r).flatten())
        center = torch.searchsorted(self.ptr, pos, right=True) - 1
        return center, self.nbr[pos], self.eid[pos], self.rev[pos]


def sample_subgraph(target_eids, src, dst, csr, fanouts=(15, 10), gen=None):
    """2-hop neighborhood around the accounts of the target transactions.

    Returns local node count, message edge_index (neighbor -> center), the
    transaction id and direction of every message edge, and the local ids of
    each target transaction's sender and receiver.
    """
    t_src, t_dst = src[target_eids], dst[target_eids]
    frontier = torch.unique(torch.cat([t_src, t_dst]))
    seen = frontier
    centers, nbrs, eids, revs = [], [], [], []
    for k in fanouts:
        c, nb, e, r = csr.sample(frontier, k, gen)
        centers.append(c); nbrs.append(nb); eids.append(e); revs.append(r)
        new = torch.unique(nb)
        frontier = new[~torch.isin(new, seen)]
        seen = torch.cat([seen, frontier])
    centers, nbrs = torch.cat(centers), torch.cat(nbrs)
    eids, revs = torch.cat(eids), torch.cat(revs)

    nodes = torch.unique(torch.cat([seen, nbrs]))  # sorted global ids
    edge_index = torch.stack([torch.searchsorted(nodes, nbrs), torch.searchsorted(nodes, centers)])
    return (len(nodes), edge_index, eids, revs,
            torch.searchsorted(nodes, t_src), torch.searchsorted(nodes, t_dst))
