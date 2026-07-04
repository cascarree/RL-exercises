"""
Dyna-PPO vs PPO: Sample Efficiency Comparison
Compare Dyna-PPO (use_model=True) vs plain PPO (use_model=False)
on CartPole-v1 with 15k real environment steps.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
NUM_SEEDS = 5
TOTAL_STEPS = 15000
EVAL_INTERVAL = 500
EVAL_EPISODES = 5

PPO_COMMON = {
    "lr_actor": 5e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 4,
    "batch_size": 64,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "hidden_size": 128,
}

DYNA_SPECIFIC = {
    "model_lr": 1e-3,
    "model_epochs": 5,
    "model_batch_size": 64,
    "imag_horizon": 5,
    "imag_batches": 20,
    "max_buffer_size": 100000,
}


def run_experiment(use_model: bool, seed: int) -> pd.DataFrame:
    """Run a single experiment and return eval curve."""
    from rl_exercises.week_9.dyna_ppo import DynaPPOAgent, set_seed

    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DynaPPOAgent(
        env,
        use_model=use_model,
        **PPO_COMMON,
        **DYNA_SPECIFIC,
        seed=seed,
    )

    eval_env = gym.make("CartPole-v1")
    records = []
    real_steps = 0

    while real_steps < TOTAL_STEPS:
        state, _ = env.reset(seed=seed)
        done = False
        trajectory = []

        while not done and real_steps < TOTAL_STEPS:
            action, logp, ent, val = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append(
                (state, action, logp, ent, reward, float(done), next_state)
            )
            state = next_state
            real_steps += 1

            if real_steps % EVAL_INTERVAL == 0:
                mean_r, std_r = agent.evaluate(eval_env, num_episodes=EVAL_EPISODES)
                records.append(
                    {"step": real_steps, "return": mean_r, "std": std_r, "seed": seed}
                )

        agent.update(trajectory)

        if use_model:
            agent.store_real(trajectory)
            agent.train_model()
            agent.imagine_and_update()

    env.close()
    eval_env.close()
    return pd.DataFrame(records)


def main() -> None:
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {"PPO": [], "Dyna-PPO": []}

    for seed in range(NUM_SEEDS):
        print(f"\n{'=' * 50}")
        print(f"Seed {seed + 1}/{NUM_SEEDS}")
        print(f"{'=' * 50}")

        print("\n--- PPO ---")
        df_ppo = run_experiment(use_model=False, seed=seed)
        results["PPO"].append(df_ppo)

        print("\n--- Dyna-PPO ---")
        df_dyna = run_experiment(use_model=True, seed=seed)
        results["Dyna-PPO"].append(df_dyna)

    # Aggregate across seeds
    steps = np.arange(0, TOTAL_STEPS + 1, EVAL_INTERVAL)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, dfs in results.items():
        # Interpolate to common steps
        interp_returns = []
        for df in dfs:
            interp = np.interp(steps, df["step"].values, df["return"].values)
            interp_returns.append(interp)
        stacked = np.stack(interp_returns)
        mean = stacked.mean(axis=0)
        se = stacked.std(axis=0, ddof=0) / np.sqrt(stacked.shape[0])

        color = "#1f77b4" if algo_name == "PPO" else "#ff7f0e"
        ax.plot(steps, mean, label=algo_name, linewidth=2, color=color)
        ax.fill_between(steps, mean - se, mean + se, alpha=0.2, color=color)

    # Find 80% of final PPO return for Dyna-PPO speedup calculation
    ppo_final = np.mean([df["return"].iloc[-5:].mean() for df in results["PPO"]])
    dyna_final = np.mean([df["return"].iloc[-5:].mean() for df in results["Dyna-PPO"]])
    ppo_80 = 0.8 * ppo_final

    # Find where each crosses 80% of PPO's final
    ppo_mean = np.mean(
        [
            np.interp(steps, df["step"].values, df["return"].values)
            for df in results["PPO"]
        ],
        axis=0,
    )
    dyna_mean = np.mean(
        [
            np.interp(steps, df["step"].values, df["return"].values)
            for df in results["Dyna-PPO"]
        ],
        axis=0,
    )

    ppo_80_step = steps[np.argmax(ppo_mean >= ppo_80)]
    dyna_80_step = steps[np.argmax(dyna_mean >= ppo_80)]

    ax.axhline(
        y=ppo_80,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"80% of PPO final ({ppo_80:.0f})",
    )
    ax.axvline(x=ppo_80_step, color="#1f77b4", linestyle=":", alpha=0.5)
    ax.axvline(x=dyna_80_step, color="#ff7f0e", linestyle=":", alpha=0.5)

    ax.set_xlabel("Real Environment Steps")
    ax.set_ylabel("Average Return")
    ax.set_title(f"Dyna-PPO vs PPO on CartPole-v1 ({NUM_SEEDS} seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "dyna_ppo_vs_ppo.png", dpi=150)
    plt.close(fig)

    # Print summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"PPO final return:      {ppo_final:.1f}")
    print(f"Dyna-PPO final return: {dyna_final:.1f}")
    print(f"80% of PPO final:      {ppo_80:.1f}")
    print(f"PPO reached 80% at:    step {ppo_80_step}")
    print(f"Dyna-PPO reached 80%:  step {dyna_80_step}")
    speedup = ppo_80_step - dyna_80_step
    print(f"Dyna-PPO reached 80% {speedup} steps sooner")

    # Early performance comparison (first 3000 steps)
    early_ppo = ppo_mean[: np.argmax(steps >= 3000) + 1].mean()
    early_dyna = dyna_mean[: np.argmax(steps >= 3000) + 1].mean()
    print("\nEarly performance (first 3k steps):")
    print(f"  PPO mean return:      {early_ppo:.1f}")
    print(f"  Dyna-PPO mean return: {early_dyna:.1f}")
    if early_dyna < early_ppo:
        print("  -> Model learning penalty: Dyna-PPO underperforms early")
    else:
        print("  -> No model learning penalty observed")

    print(f"\nPlot saved to {PLOTS_DIR / 'dyna_ppo_vs_ppo.png'}")


if __name__ == "__main__":
    main()
