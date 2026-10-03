"""Representational analysis: how many states can the questions of a depth-d symmetric tree
distinguish, and what is the rank of the outcome matrix (the linear-PSR dimension needed)?"""
from common import *
from tdnet import *

lines = ["| world | states | depth | nodes | distinguishable states | rank of outcome matrix |", "|---|---|---|---|---|---|"]
for name, env in [("ring5", ring_world(5)), ("ring8", ring_world(8)), ("cycle6", cycle_world(6)),
                  ("room4", gridworld(MAPS["room4"])), ("ijcai26 (bit-to-bit)", gridworld(MAPS["ijcai26"])),
                  ("tworooms", gridworld(MAPS["tworooms"])), ("maze12", gridworld(MAPS["maze12"]))]:
    S = env.n_states
    for d in range(1, 9):
        q = symmetric_tree(env.n_actions, d) if env.n_actions > 1 else chain(d)
        V = env.node_values(q.action_sequences())
        distinct = len(np.unique(np.round(V.T, 6), axis=0)); rank = np.linalg.matrix_rank(V, tol=1e-6)
        lines.append(f"| {name} | {S} | {d} | {q.n_active} | {distinct} | {rank} |")
        if distinct == S and rank == S:
            break
out = "\n".join(lines)
print(out)
open(os.path.join(DATA, "rank_table.md"), "w").write(out + "\n")
