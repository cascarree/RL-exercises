"""
Level 1: How Well Does HPO Generalize in RL?
Random search over PPO hyperparameters on CartPole-v1.
Test generalization across seeds and environments.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
N_TRIALS = 20
TRAIN_STEPS = 8000
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


def evaluate_config(cfg: dict, env_id: str, seed: int) -> float:
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
    agent.train(TRAIN_STEPS, eval_interval=EVAL_STEPS, eval_episodes=EVAL_EPISODES)

    eval_env = gym.make(env_id)
    returns = []
    for _ in range(10):
        state, _ = eval_env.reset(seed=seed + 1000)
        done = False
        total = 0.0
        while not done:
            action, _, _, _ = agent.predict(state)
            state, r, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total += r
        returns.append(total)
    env.close()
    eval_env.close()
    return float(np.mean(returns))


def main() -> None:
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Level 1: HPO Generalization")
    print("=" * 60)

    # Phase 1: Random search on CartPole-v1, seed=0
    print(f"\n--- Phase 1: Random search ({N_TRIALS} trials, seed=0) ---")
    trial_results = []
    for i in range(N_TRIALS):
        cfg = sample_config()
        score = evaluate_config(cfg, "CartPole-v1", seed=0)
        trial_results.append((cfg, score))
        print(
            f"  Trial {i + 1:2d}/{N_TRIALS}: return={score:6.1f} | "
            f"lr={cfg['lr_actor']:.4f} clip={cfg['clip_eps']:.2f} epochs={cfg['epochs']}"
        )

    # Sort by score
    trial_results.sort(key=lambda x: x[1], reverse=True)
    best_cfg = trial_results[0][0]
    best_score = trial_results[0][1]
    print(f"\nBest config: return={best_score:.1f}")
    print(
        f"  lr_actor={best_cfg['lr_actor']:.4f}, clip_eps={best_cfg['clip_eps']:.2f}, "
        f"epochs={best_cfg['epochs']}, batch_size={best_cfg['batch_size']}, "
        f"ent_coef={best_cfg['ent_coef']:.4f}, hidden_size={best_cfg['hidden_size']}"
    )

    # Phase 2: Generalization across seeds
    print("\n--- Phase 2: Generalization across seeds ---")
    seed_returns = []
    for seed in range(5):
        r = evaluate_config(best_cfg, "CartPole-v1", seed=seed)
        seed_returns.append(r)
        print(f"  seed={seed}: return={r:.1f}")

    # Phase 3: Generalization across environments
    print("\n--- Phase 3: Generalization across environments ---")
    envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    env_returns = {}
    for env_id in envs:
        r = evaluate_config(best_cfg, env_id, seed=0)
        env_returns[env_id] = r
        print(f"  {env_id}: return={r:.1f}")

    # Phase 4: Optimize separately on each environment, compare
    print("\n--- Phase 4: Per-environment optimization ---")
    per_env_best = {}
    for env_id in envs:
        best_r, best_c = -float("inf"), None
        for _ in range(N_TRIALS):
            cfg = sample_config()
            r = evaluate_config(cfg, env_id, seed=0)
            if r > best_r:
                best_r, best_c = r, cfg
        per_env_best[env_id] = (best_r, best_c)
        print(f"  {env_id} best: return={best_r:.1f}")

    # Cross-evaluate: test each env's best config on all envs
    print("\n--- Phase 5: Cross-environment generalization ---")
    cross_scores = {}
    for train_env, (_score, cfg) in per_env_best.items():
        cross_scores[train_env] = {}
        for test_env in envs:
            r = evaluate_config(cfg, test_env, seed=0)
            cross_scores[train_env][test_env] = r
            print(f"  Trained on {train_env}, tested on {test_env}: {r:.1f}")

    # --- Plots ---
    # Plot 1: Trial scores
    fig, ax = plt.subplots(figsize=(10, 5))
    scores = [s for _, s in trial_results]
    ax.bar(range(len(scores)), scores, color="#1f77b4", alpha=0.7)
    ax.axhline(
        y=best_score, color="#d1495b", linestyle="--", label=f"Best: {best_score:.1f}"
    )
    ax.set_xlabel("Trial (sorted by score)")
    ax.set_ylabel("Return")
    ax.set_title(f"Level 1: Random Search Results ({N_TRIALS} trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "hpo_trial_scores.png", dpi=150)
    plt.close()

    # Plot 2: Seed generalization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(seed_returns)), seed_returns, color="#2a9d8f", edgecolor="black")
    ax.axhline(
        y=best_score,
        color="#d1495b",
        linestyle="--",
        label=f"Train score (seed=0): {best_score:.1f}",
    )
    ax.set_xticks(range(len(seed_returns)))
    ax.set_xticklabels([f"seed={i}" for i in range(5)])
    ax.set_ylabel("Return")
    ax.set_title("Level 1: Generalization Across Seeds")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "seed_generalization.png", dpi=150)
    plt.close()

    # Plot 3: Cross-env generalization heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = np.array([[cross_scores[t][ts] for ts in envs] for t in envs])
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=45, ha="right")
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels(envs)
    ax.set_xlabel("Test Environment")
    ax.set_ylabel("Train Environment")
    ax.set_title("Level 1: Cross-Environment HPO Generalization")
    for i in range(len(envs)):
        for j in range(len(envs)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Return")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "cross_env_generalization.png", dpi=150)
    plt.close()

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
