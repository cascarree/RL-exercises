"""
Level 2: Multi-fidelity in RL
Compare full-budget HPO vs early-stopping multi-fidelity.
Show when early success doesn't translate to final performance.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
N_TRIALS = 15
FULL_BUDGET = 10000
CHECKPOINT_BUDGET = 3000
EVAL_STEPS = 500
EVAL_EPISODES = 3

SEARCH_SPACE = {
    "lr_actor": (1e-4, 1e-2, "log"),
    "lr_critic": (1e-4, 1e-2, "log"),
    "clip_eps": (0.1, 0.3, "linear"),
    "epochs": (2, 8, "int"),
    "batch_size": (32, 128, "int"),
    "ent_coef": (0.001, 0.05, "log"),
    "hidden_size": (32, 128, "int"),
}


def sample_config() -> dict:
    cfg = {}
    for name, (lo, hi, kind) in SEARCH_SPACE.items():
        if kind == "log":
            cfg[name] = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
        elif kind == "linear":
            cfg[name] = np.random.uniform(lo, hi)
        elif kind == "int":
            cfg[name] = int(np.random.randint(lo, hi + 1))
    return cfg


def train_and_eval(
    cfg: dict, env_id: str, seed: int, total_steps: int
) -> tuple[float, float]:
    """Train for total_steps, return (final_return, checkpoint_return)."""
    from rl_exercises.week_6.ppo import PPOAgent, set_seed

    env = gym.make(env_id)
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = PPOAgent(
        env,
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        clip_eps=cfg["clip_eps"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        ent_coef=cfg["ent_coef"],
        hidden_size=int(cfg["hidden_size"]),
        seed=seed,
    )

    eval_env = gym.make(env_id)

    # Train with checkpoint eval at CHECKPOINT_BUDGET
    checkpoint_return = None
    step_count = 0

    state, _ = env.reset(seed=seed)
    done = False
    traj = []

    while step_count < total_steps:
        action, logp, ent, val = agent.predict(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        traj.append((state, action, logp, ent, reward, float(done), next_state))
        state = next_state
        step_count += 1

        if done:
            state, _ = env.reset(seed=seed)
            done = False
            traj = []

        if step_count >= CHECKPOINT_BUDGET and checkpoint_return is None:
            # Evaluate at checkpoint
            rets = []
            for _ in range(EVAL_EPISODES):
                s, _ = eval_env.reset(seed=seed + 500)
                d = False
                total = 0.0
                while not d:
                    a, _, _, _ = agent.predict(s)
                    s, r, t, tr, _ = eval_env.step(a)
                    d = t or tr
                    total += r
                rets.append(total)
            checkpoint_return = float(np.mean(rets))

        if step_count % 2000 == 0 and step_count < total_steps and traj:
            agent.update(traj)
            traj = []

    # Final PPO update
    if traj:
        agent.update(traj)

    # Final evaluation
    rets = []
    for _ in range(10):
        s, _ = eval_env.reset(seed=seed + 1000)
        d = False
        total = 0.0
        while not d:
            a, _, _, _ = agent.predict(s)
            s, r, t, tr, _ = eval_env.step(a)
            d = t or tr
            total += r
        rets.append(total)

    env.close()
    eval_env.close()
    return float(np.mean(rets)), checkpoint_return or 0.0


def main() -> None:
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Level 2: Multi-fidelity in RL")
    print("=" * 60)

    # Generate all configs upfront
    configs = [sample_config() for _ in range(N_TRIALS)]

    # --- Full budget HPO ---
    print(f"\n--- Full budget HPO ({FULL_BUDGET} steps, {N_TRIALS} trials) ---")
    full_results = []
    for i, cfg in enumerate(configs):
        final_r, cp_r = train_and_eval(
            cfg, "CartPole-v1", seed=0, total_steps=FULL_BUDGET
        )
        full_results.append(
            {"trial": i, "final": final_r, "checkpoint": cp_r, "cfg": cfg}
        )
        print(f"  Trial {i + 1:2d}: final={final_r:6.1f} cp={cp_r:6.1f}")

    # --- Multi-fidelity: train only good checkpoints to full budget ---
    print(
        f"\n--- Multi-fidelity: train checkpoint={CHECKPOINT_BUDGET} steps, promote top 50% ---"
    )
    # Phase 1: Train all to checkpoint
    cp_results = []
    for i, cfg in enumerate(configs):
        final_r, cp_r = train_and_eval(
            cfg, "CartPole-v1", seed=0, total_steps=CHECKPOINT_BUDGET
        )
        cp_results.append({"trial": i, "checkpoint": cp_r, "cfg": cfg})
        print(f"  Trial {i + 1:2d}: checkpoint={cp_r:6.1f}")

    # Phase 2: Promote top 50% to full budget
    cp_results.sort(key=lambda x: x["checkpoint"], reverse=True)
    n_promote = len(cp_results) // 2
    promoted = cp_results[:n_promote]
    print(f"\n  Promoting top {n_promote} to full budget...")

    mf_results = []
    for item in promoted:
        final_r, _ = train_and_eval(
            item["cfg"], "CartPole-v1", seed=0, total_steps=FULL_BUDGET
        )
        mf_results.append(
            {
                "trial": item["trial"],
                "final": final_r,
                "checkpoint": item["checkpoint"],
            }
        )
        print(
            f"  Trial {item['trial'] + 1:2d}: cp={item['checkpoint']:6.1f} -> final={final_r:6.1f}"
        )

    # --- Analyze: early success vs final performance ---
    full_finals = sorted([r["final"] for r in full_results], reverse=True)
    mf_finals = sorted([r["final"] for r in mf_results], reverse=True)

    print("\n--- Results ---")
    print(f"Full budget best:       {full_finals[0]:.1f}")
    print(f"Full budget median:     {np.median(full_finals):.1f}")
    if mf_finals:
        print(f"Multi-fidelity best:    {mf_finals[0]:.1f}")
        print(f"Multi-fidelity median:  {np.median(mf_finals):.1f}")

    # --- Plots ---
    # Plot 1: Checkpoint vs Final correlation (full budget)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in full_results:
        color = "#2a9d8f" if r["final"] >= np.median(full_finals) else "#d1495b"
        ax.scatter(
            r["checkpoint"], r["final"], c=color, s=80, alpha=0.7, edgecolors="black"
        )
    ax.set_xlabel("Return at Checkpoint (3k steps)")
    ax.set_ylabel("Final Return (10k steps)")
    ax.set_title("Level 2: Checkpoint vs Final Performance")
    ax.grid(True, alpha=0.3)
    # Add correlation
    cps = [r["checkpoint"] for r in full_results]
    fs = [r["final"] for r in full_results]
    corr = np.corrcoef(cps, fs)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "checkpoint_vs_final.png", dpi=150)
    plt.close()

    # Plot 2: Full budget vs Multi-fidelity comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(full_finals))
    ax.bar(x - 0.2, full_finals, 0.4, label="Full budget", color="#1f77b4", alpha=0.7)
    ax.bar(
        x[: len(mf_finals)] + 0.2,
        mf_finals,
        0.4,
        label="Multi-fidelity",
        color="#ff7f0e",
        alpha=0.7,
    )
    ax.set_xlabel("Trial (sorted by return)")
    ax.set_ylabel("Final Return")
    ax.set_title("Level 2: Full Budget vs Multi-fidelity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "full_vs_multifidelity.png", dpi=150)
    plt.close()

    # Plot 3: Promotion analysis - checkpoint ranking vs final ranking
    fig, ax = plt.subplots(figsize=(8, 6))
    cp_rank = list(range(len(cp_results)))
    final_rank = []
    final_map = {r["trial"]: r["final"] for r in full_results}
    for item in cp_results:
        final_rank.append(
            sorted(full_finals, reverse=True).index(final_map[item["trial"]])
        )
    promoted_idx = list(range(n_promote))
    not_promoted_idx = list(range(n_promote, len(cp_results)))
    ax.scatter(
        [cp_rank[i] for i in promoted_idx],
        [final_rank[i] for i in promoted_idx],
        c="#2a9d8f",
        s=80,
        label="Promoted",
        edgecolors="black",
    )
    ax.scatter(
        [cp_rank[i] for i in not_promoted_idx],
        [final_rank[i] for i in not_promoted_idx],
        c="#d1495b",
        s=80,
        label="Pruned",
        edgecolors="black",
        alpha=0.5,
    )
    ax.set_xlabel("Checkpoint Rank (lower = better at checkpoint)")
    ax.set_ylabel("Final Rank (lower = better final)")
    ax.set_title("Level 2: Does Early Rank Predict Final Rank?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.plot(
        [0, len(cp_results) - 1],
        [0, len(cp_results) - 1],
        "k--",
        alpha=0.3,
        label="Perfect correlation",
    )
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "rank_correlation.png", dpi=150)
    plt.close()

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
