"""
Level 2: Statistical Testing
Compare DQN vs REINFORCE on CartPole-v1 using 10 seeds each.
Uses paired t-test, Wilcoxon signed-rank, and Mann-Whitney U tests.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
NUM_SEEDS = 10
NUM_FRAMES = 5000
EVAL_INTERVAL = 200
SIGNIFICANCE_LEVEL = 0.05

DQN_CONFIG = {
    "hidden_dim": 64,
    "depth": 2,
    "buffer_capacity": 5000,
    "batch_size": 32,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay": 500,
    "target_update_freq": 500,
}

REINFORCE_CONFIG = {
    "lr": 1e-2,
    "gamma": 0.99,
    "hidden_size": 128,
}


def run_dqn_seed(seed: int, num_frames: int) -> pd.DataFrame:
    from rl_exercises.week_4.dqn import DQNAgent, set_seed

    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DQNAgent(
        env,
        buffer_capacity=DQN_CONFIG["buffer_capacity"],
        batch_size=DQN_CONFIG["batch_size"],
        lr=DQN_CONFIG["lr"],
        gamma=DQN_CONFIG["gamma"],
        epsilon_start=DQN_CONFIG["epsilon_start"],
        epsilon_final=DQN_CONFIG["epsilon_final"],
        epsilon_decay=DQN_CONFIG["epsilon_decay"],
        target_update_freq=DQN_CONFIG["target_update_freq"],
        hidden_dim=DQN_CONFIG["hidden_dim"],
        depth=DQN_CONFIG["depth"],
        seed=seed,
    )

    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    frame = 0
    eval_frames: list[int] = []
    eval_rewards: list[float] = []
    episode_buffer: list[float] = []

    while frame < num_frames:
        action = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.add(state, action, reward, next_state, done, {})
        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)
        episode_reward += float(reward)
        frame += 1
        state = next_state
        if done:
            episode_buffer.append(episode_reward)
            state, _ = env.reset(seed=seed)
            episode_reward = 0.0
        if frame % EVAL_INTERVAL == 0 and episode_buffer:
            eval_frames.append(frame)
            eval_rewards.append(float(np.mean(episode_buffer[-10:])))

    env.close()
    return pd.DataFrame({"frame": eval_frames, "reward": eval_rewards, "seed": seed})


def run_reinforce_seed(seed: int, num_frames: int) -> pd.DataFrame:
    from rl_exercises.week_5.policy_gradient import REINFORCEAgent, set_seed

    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = REINFORCEAgent(
        env=env,
        lr=REINFORCE_CONFIG["lr"],
        gamma=REINFORCE_CONFIG["gamma"],
        seed=seed,
        hidden_size=REINFORCE_CONFIG["hidden_size"],
    )

    eval_frames: list[int] = []
    eval_rewards: list[float] = []
    eval_env = gym.make("CartPole-v1")
    total_frames = 0
    episode_count = 0

    while total_frames < num_frames:
        state, _ = env.reset(seed=seed)
        done = False
        batch = []
        while not done:
            action, info = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            batch.append((state, action, float(reward), next_state, done, info))
            state = next_state
            total_frames += 1
            if total_frames >= num_frames:
                break
        agent.update_agent(batch)
        episode_count += 1
        if episode_count % 5 == 0 and total_frames < num_frames:
            mean_ret, _ = agent.evaluate(eval_env, num_episodes=5)
            eval_frames.append(total_frames)
            eval_rewards.append(mean_ret)

    eval_env.close()
    env.close()
    return pd.DataFrame({"frame": eval_frames, "reward": eval_rewards, "seed": seed})


def align(dfs: list[pd.DataFrame], grid: np.ndarray) -> np.ndarray:
    aligned = []
    for df in dfs:
        f = df["frame"].to_numpy(dtype=float)
        r = df["reward"].to_numpy(dtype=float)
        aligned.append(np.interp(grid, f, r, left=r[0], right=r[-1]))
    return np.stack(aligned, axis=0).copy()


def run_statistical_tests(dqn_finals: np.ndarray, reinforce_finals: np.ndarray) -> dict:
    t_stat, t_pval = sp_stats.ttest_rel(dqn_finals, reinforce_finals)
    w_stat, w_pval = sp_stats.wilcoxon(dqn_finals, reinforce_finals)
    u_stat, u_pval = sp_stats.mannwhitneyu(
        dqn_finals, reinforce_finals, alternative="two-sided"
    )

    pooled_std = np.sqrt(
        (
            (len(dqn_finals) - 1) * dqn_finals.std(ddof=1) ** 2
            + (len(reinforce_finals) - 1) * reinforce_finals.std(ddof=1) ** 2
        )
        / (len(dqn_finals) + len(reinforce_finals) - 2)
    )
    cohens_d = (
        (dqn_finals.mean() - reinforce_finals.mean()) / pooled_std
        if pooled_std > 0
        else 0.0
    )

    return {
        "paired_t_stat": t_stat,
        "paired_t_pval": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval,
        "mannwhitney_u_stat": u_stat,
        "mannwhitney_u_pval": u_pval,
        "cohens_d": cohens_d,
        "dqn_mean": float(dqn_finals.mean()),
        "dqn_std": float(dqn_finals.std(ddof=1)),
        "reinforce_mean": float(reinforce_finals.mean()),
        "reinforce_std": float(reinforce_finals.std(ddof=1)),
    }


def main() -> None:
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(EVAL_INTERVAL, NUM_FRAMES, 50)

    # Run all seeds
    dqn_dfs, dqn_finals = [], []
    reinforce_dfs, reinforce_finals = [], []

    print(f"\nRunning DQN ({NUM_SEEDS} seeds)...")
    for seed in range(NUM_SEEDS):
        print(f"  seed={seed}")
        df = run_dqn_seed(seed, NUM_FRAMES)
        dqn_dfs.append(df)
        dqn_finals.append(float(df["reward"].tail(5).mean()) if len(df) >= 5 else 0.0)

    print(f"\nRunning REINFORCE ({NUM_SEEDS} seeds)...")
    for seed in range(NUM_SEEDS):
        print(f"  seed={seed}")
        df = run_reinforce_seed(seed, NUM_FRAMES)
        reinforce_dfs.append(df)
        reinforce_finals.append(
            float(df["reward"].tail(5).mean()) if len(df) >= 5 else 0.0
        )

    dqn_finals = np.array(dqn_finals)
    reinforce_finals = np.array(reinforce_finals)

    # Align to grid
    dqn_matrix = align(dqn_dfs, grid)
    reinforce_matrix = align(reinforce_dfs, grid)

    # Plot 1: Learning curves (mean ± SE)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, matrix, color in [
        ("DQN", dqn_matrix, "#1f77b4"),
        ("REINFORCE", reinforce_matrix, "#ff7f0e"),
    ]:
        mean = matrix.mean(axis=0)
        se = matrix.std(axis=0, ddof=0) / np.sqrt(matrix.shape[0])
        ax.plot(grid, mean, label=f"{name} mean", linewidth=2, color=color)
        ax.fill_between(
            grid, mean - se, mean + se, alpha=0.2, color=color, label=f"{name} ± SE"
        )
    ax.set_xlabel("Frames")
    ax.set_ylabel("Reward")
    ax.set_title("DQN vs REINFORCE on CartPole-v1 (10 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "l2_comparison.png", dpi=150)
    plt.close(fig)

    # Statistical tests
    results = run_statistical_tests(dqn_finals, reinforce_finals)

    print(f"\n{'=' * 60}")
    print("STATISTICAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"DQN:       {results['dqn_mean']:.2f} +/- {results['dqn_std']:.2f}")
    print(
        f"REINFORCE: {results['reinforce_mean']:.2f} +/- {results['reinforce_std']:.2f}"
    )
    print(
        f"\nPaired t-test:      stat={results['paired_t_stat']:.3f}, p={results['paired_t_pval']:.4f} {'*' if results['paired_t_pval'] < SIGNIFICANCE_LEVEL else 'ns'}"
    )
    print(
        f"Wilcoxon signed-rank: stat={results['wilcoxon_stat']:.3f}, p={results['wilcoxon_pval']:.4f} {'*' if results['wilcoxon_pval'] < SIGNIFICANCE_LEVEL else 'ns'}"
    )
    print(
        f"Mann-Whitney U:      stat={results['mannwhitney_u_stat']:.3f}, p={results['mannwhitney_u_pval']:.4f} {'*' if results['mannwhitney_u_pval'] < SIGNIFICANCE_LEVEL else 'ns'}"
    )
    print(f"Cohen's d:           {results['cohens_d']:.3f}")
    print(f"\nSignificance level: {SIGNIFICANCE_LEVEL}")
    print(f"\nPlot saved to {PLOTS_DIR / 'l2_comparison.png'}")


if __name__ == "__main__":
    main()
