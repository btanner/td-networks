"""Regenerate every figure from the saved JSON results (no re-running)."""
from common import *
import os

def C(R, k, label): return (label, R[k]["step"], R[k]["rmse"])

if os.path.exists(os.path.join(DATA, "exp_paper.json")):
    R = load_json("exp_paper")
    plot_curves([C(R, f"ring5_lam{l}", f"lambda={l}") for l in [0.0, 0.25, 0.5, 0.75, 1.0]],
                "5-state ring world, symmetric depth-3 question net (16 streams)", "paper_ring5_lambda.png", ref=0.05, ymax=0.5)
    plot_curves([C(R, "ring5_lam1.0", "weight set per (a,o), linear"), C(R, "ring5_paperfeat_linear", "paper features, single matrix"),
                 C(R, "ring5_paperfeat_mlp", "paper features, MLP-32")],
                "5-state ring: the answer network must gate on the action", "paper_ring5_representation.png", ref=0.05, ymax=0.5)
    plot_curves([C(R, "ring5_lam1_B1", "lambda=1"), C(R, "ring5_lam0_B1", "lambda=0")],
                "5-state ring world, single stream", "paper_ring5_single_stream.png", ref=0.05, ymax=0.5)
    plot_curves([C(R, f"ring8_lam{l}", f"lambda={l}") for l in [0.0, 0.5, 0.9, 1.0]],
                "8-state ring world, sparse 16-node question net (16 streams)", "paper_ring8_lambda.png", ref=0.05, ymax=0.5)
    plot_curves([C(R, f"cycle6_lam{l}", f"lambda={l}") for l in [0.0, 0.25, 0.5, 0.75, 1.0]],
                "6-state cycle world, 5-chain question net (16 streams)", "paper_cycle6_lambda.png", ref=0.05, ymax=0.5)
    plot_curves([C(R, f"cycle6_lam0_hf{k}", f"history {k}") for k in [1, 2, 3, 4]],
                "6-state cycle world, TD(0) with history features (IJCAI'05)", "paper_cycle6_history.png", ref=0.05, ymax=0.5)

if os.path.exists(os.path.join(DATA, "exp_grid.json")):
    R = load_json("exp_grid")
    plot_curves([C(R, f"d{d}_hf{h}_lam1", f"depth {d}, history {h}") for d in [2, 3, 4, 5] for h in [0, 3]],
                "Bit-to-bit gridworld (104 states): depth x history, lambda=1, alpha=0.5", "grid_depth_history.png", ymax=0.5)
    plot_curves([C(R, f"d4_hf2_lam{l}", f"lambda={l}") for l in [0.0, 0.5, 1.0]],
                "Bit-to-bit gridworld: effect of lambda (depth 4, history 2, alpha=0.5)", "grid_lambda.png", ymax=0.5)
    plot_curves([C(R, "d4_hf0_lam1", "weight set per (a,o)"), C(R, "d4_hf2_lam1", "weight set per (a,o) + history-2 features"),
                 C(R, "d4_hw2", "weight set per 2-step history (16 sets)"), C(R, "d4_hf2_mlp", "history-2 features, MLP-64, SGD (diverged)"),
                 C(R, "d4_hf2_mlp_adam", "history-2 features, MLP-64 + Adam")],
                "Bit-to-bit gridworld: answer-network variants (depth 4, lambda=1)", "grid_answer_nets.png", ymax=0.5)

if os.path.exists(os.path.join(DATA, "exp_alpha.json")):
    R = load_json("exp_alpha")
    plot_curves([C(R, k, f"depth {k[1]}, history {k.split('_hf')[1][0]}, alpha {k.split('_a')[1]}") for k in R if k.startswith("d") and "_a0" in k and "lam" not in k],
                "Bit-to-bit gridworld: deeper fixed trees at smaller step sizes", "grid_alpha.png", ymax=0.5)
    plot_curves([C(R, f"d4_hf2_lam{l}_a0.1", f"lambda {l}") for l in ["0", "0.5", "1"]],
                "Bit-to-bit gridworld: lambda at alpha 0.1 (depth 4, history 2)", "grid_lambda_alpha0.1.png", ymax=0.5)

if os.path.exists(os.path.join(DATA, "exp_big.json")):
    R = load_json("exp_big")
    plot_curves([C(R, f"{m}_1bit_d4_hf3", f"{m} ({R[f'{m}_1bit_d4_hf3']['S']} states)") for m in ["ijcai26", "tworooms", "maze12"]],
                "Larger 1-bit maps: depth-4 tree, 3-step history, lambda=1", "big_maps_1bit.png", ymax=0.5)
    plot_curves([C(R, "ijcai26_3bit_d2", "3 bits, F/R, depth 2"), C(R, "ijcai26_3bit_d3", "3 bits, F/R, depth 3"),
                 C(R, "ijcai26_3bit_4act_d2", "3 bits, F/R/L/B, depth 2"), C(R, "ijcai26_1bit_4act_d3", "1 bit, F/R/L/B, depth 3")],
                "26-cell map: multi-bit sensors and four actions (lambda=1)", "big_obs_actions.png", ymax=0.5)
    cur = [C(R, "ijcai26_1bit_d3_hf0", "deterministic, linear")]
    for env, lab in [("grid26c2a1b_slip0.1", "slip 0.1"), ("grid26c2a1b_noisy", "noisy sensor"), ("grid26c2a1b_slip0.1_noisy", "slip + noisy")]:
        cur.append(C(R, f"{env}_d3_hf2_lam1.0", f"{lab}, linear")); cur.append(C(R, f"{env}_d3_hf2_mlp", f"{lab}, MLP-64 + Adam"))
    plot_curves(cur, "26-cell map, stochastic motion / noisy sensor (depth 3, history 2, lambda=1)", "big_stochastic.png", ymax=0.5)

if os.path.exists(os.path.join(DATA, "exp_history_baseline.json")):
    R = load_json("exp_history_baseline")
    for env in ["ring5", "ring8", "cycle6", "bit2bit104"]:
        ks = sorted({int(k.split("_k")[1]) for k in R if k.startswith(env + "_k")})
        plot_curves([C(R, f"{env}_k{k}", f"history k={k}") for k in ks], f"{env}: history-only next-observation predictors",
                    f"history_{env}.png", ylabel="oracle RMSE (next-observation predictions)", ymax=0.5)
print("figures regenerated")
