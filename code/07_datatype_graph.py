"""07_datatype_graph.py -- graph/network data: build a scale-free network + layout visualization."""  # module comment
import numpy as np                                                    # numerical computing
import pandas as pd                                                   # export the edge table and node table
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, DAT, save, dump_csv                          # shared helpers

N, M = 60, 2                                                          # 60 nodes; each added node connects 2 edges (BA model)
edges = []                                                            # initialize the edge list
targets = [0, 1]                                                      # initial core: nodes 0 and 1 are already connected
edges.append((0, 1))                                                  # record the core edge
for v in range(2, N):                                                 # add nodes 2..N-1 one by one (preferential attachment)
    deg = np.zeros(v)                                                 # degree count over the existing v nodes
    for a, b in edges:                                                # accumulate degrees over the existing edges
        deg[a] += 1                                                   # source degree +1
        deg[b] += 1                                                   # target degree +1
    prob = (deg + 1) / (deg + 1).sum()                                # connection probability proportional to degree (+1 smoothing to avoid zero)
    chosen = RNG.choice(v, size=M, replace=False, p=prob)             # draw M neighbors by probability without replacement
    for c in chosen:                                                  # loop over the chosen neighbors
        edges.append((int(c), v))                                     # add a new edge (yielding a "rich get richer" structure)
adj = np.zeros((N, N))                                                # initialize the adjacency matrix
for a, b in edges:                                                    # fill the adjacency matrix from the edge list
    adj[a, b] = adj[b, a] = 1                                         # undirected graph: set the matrix symmetrically to 1
deg = adj.sum(axis=1)                                                 # each node's degree = the row sum

# ---------------------------------------------------------------- force-directed layout (a simplified Fruchterman-Reingold)
pos = RNG.uniform(0, 1, (N, 2))                                       # randomly initialize the node coordinates
for it in range(120):                                                 # run 120 iterations to let the layout converge
    delta = pos[:, None, :] - pos[None, :, :]                         # displacement vectors between every pair of points
    dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9                       # pairwise distance (add a tiny value to avoid divide-by-zero)
    rep = (delta / dist[..., None]) / dist[..., None] ** 2            # repulsion between every pair of points (Coulomb-like, inverse to distance squared)
    np.fill_diagonal(rep[:, :, 0], 0); np.fill_diagonal(rep[:, :, 1], 0)  # no repulsion of a point with itself
    attr = -(delta / dist[..., None]) * dist[..., None] * adj[..., None]  # attraction between adjacent points (along the connecting line)
    step = (rep.sum(1) + attr.sum(1)) * 0.002                         # net force x step size = the displacement per round
    pos = pos + np.clip(step, -0.02, 0.02)                            # clamp the per-step displacement to avoid oscillation
pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0) + 1e-9)           # normalize to [0,1] for easy plotting

dump_csv(pd.DataFrame({"src": [a for a, _ in edges],                  # edge table: source
                       "dst": [b for _, b in edges]}),                # edge table: target
         "graph_edges.csv")                                           # dump the edge table
dump_csv(pd.DataFrame({"node": range(N), "degree": deg}),             # node table: node id
         "graph_nodes.csv")                                           # dump the node table (with degrees)

# ---------------------------------------------------------------- visualization
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))                    # one row, two columns: network graph / degree distribution
for a, b in edges:                                                    # draw a line for each edge
    axes[0].plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],      # connect the coordinates of the two endpoints
                 color="#B9BFC7", lw=0.8, zorder=1)                   # thin gray lines, drawn below the nodes
axes[0].scatter(pos[:, 0], pos[:, 1], s=20 + 9 * deg,                 # node size grows with degree
                c=deg, cmap="RdPu", edgecolors="#711426", zorder=2)  # color also encodes degree, to spot hubs
axes[0].set_title("Network graph (%d nodes / %d edges, BA scale-free)" % (N, len(edges)))  # left title
axes[0].set_xticks([]); axes[0].set_yticks([])                        # hide the ticks
axes[0].grid(False)                                                   # turn off the grid
axes[0].set_aspect("equal")                                           # equal aspect to avoid stretching the graph

vals, cnt = np.unique(deg, return_counts=True)                        # compute the degree distribution (degree -> node count)
axes[1].loglog(vals, cnt, "o", color="#8A0C3C", ms=6)                 # right: degree distribution on log-log axes (roughly a straight line)
axes[1].set_title("Degree distribution (log-log: power-law feature)")  # right title
axes[1].set_xlabel("Node degree k")                                   # right x-axis meaning
axes[1].set_ylabel("Node count P(k) with degree k")                   # right y-axis meaning
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_graph.png")                                            # save the figure
