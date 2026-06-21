"""
Level 1: Seeding Analysis
Run DQN on CartPole-v1 with varying numbers of seeds (low=3, medium=10, large=30).
Compare disparate seed sets for the low count.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
EVAL_INTERVAL = 200
NUM_FRAMES = 5000

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

SEED_SETS = {
    "sequential [0,1,2]": [0, 1, 2],
    "sparse [0,100,200]": [0, 100, 200],
    "powers-of-2 [1,2,4]": [1, 2, 4],
}
MEDIUM_SEEDS = list(range(10))
LARGE_SEEDS = list(range(30))


def run_single_seed(seed: int, num_frames: int) -> pd.DataFrame:
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


def align(dfs: list[pd.DataFrame], grid: np.ndarray) -> np.ndarray:
    aligned = []
    for df in dfs:
        f = df["frame"].to_numpy(dtype=float)
        r = df["reward"].to_numpy(dtype=float)
        aligned.append(np.interp(grid, f, r, left=r[0], right=r[-1]))
    return np.stack(aligned, axis=0).copy()


def main() -> None:
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(EVAL_INTERVAL, NUM_FRAMES, 50)

    # Run all groups
    groups = {}
    for name, seeds in SEED_SETS.items():
        dfs = [run_single_seed(s, NUM_FRAMES) for s in seeds]
        groups[name] = align(dfs, grid)

    dfs_medium = [run_single_seed(s, NUM_FRAMES) for s in MEDIUM_SEEDS]
    groups["10 seeds"] = align(dfs_medium, grid)

    dfs_large = [run_single_seed(s, NUM_FRAMES) for s in LARGE_SEEDS]
    groups["30 seeds"] = align(dfs_large, grid)

    # Plot 1: All seed counts compared (mean ± SE)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d1495b", "#e8a838", "#30638e", "#2a9d8f", "#1f77b4"]
    for (name, matrix), color in zip(groups.items(), colors):
        mean = matrix.mean(axis=0)
        se = matrix.std(axis=0, ddof=0) / np.sqrt(matrix.shape[0])
        ax.plot(grid, mean, label=name, linewidth=2, color=color)
        ax.fill_between(grid, mean - se, mean + se, alpha=0.15, color=color)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mean Reward")
    ax.set_title("DQN on CartPole-v1: Effect of Seed Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "seed_count_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: Disparate seed sets (3 seeds each)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, matrix), color in zip(
        list(groups.items())[:3], ["#d1495b", "#e8a838", "#30638e"]
    ):
        mean = matrix.mean(axis=0)
        ax.plot(grid, mean, label=name, linewidth=2, color=color)
        for i in range(matrix.shape[0]):
            ax.plot(grid, matrix[i], color=color, alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Reward")
    ax.set_title("DQN on CartPole-v1: 3 Seeds, Different Seed Sets")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "disparate_seed_sets.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
