"""
Level 1.2: Model Prediction Accuracy
Evaluate dynamics model at checkpoints during training.
Measure one-step MSE and multi-step error E_k for k=1..20.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
TOTAL_STEPS = 20000
EVAL_INTERVAL = 2000
NUM_SEEDS = 3
MAX_K = 20

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


def compute_multistep_error(model, buffer, max_k=20, num_samples=200):
    """Compute E_k: average MSE after k imagined steps."""
    if len(buffer) < num_samples:
        return np.zeros(max_k)

    errors = []
    for _ in range(num_samples):
        s, a, r, s2, _ = buffer[np.random.randint(len(buffer))]
        state = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([a], dtype=torch.long)

        e_k = []
        current_state = state.clone()

        for k in range(max_k):
            a_oh = torch.zeros(1, model.fc1.in_features - current_state.shape[1])
            a_oh[0, action.item()] = 1.0

            with torch.no_grad():
                delta, _ = model(current_state, a_oh)
                next_state = current_state + delta

            # Use the model's own predictions recursively (open-loop)
            e_k.append(
                float(
                    nn.MSELoss()(
                        next_state, torch.tensor(s2, dtype=torch.float32).unsqueeze(0)
                    )
                )
            )
            current_state = next_state.detach()

        errors.append(e_k)

    return np.mean(errors, axis=0)


def run_experiment_with_model_eval(seed: int) -> dict:
    """Run Dyna-PPO and evaluate model at checkpoints."""
    from rl_exercises.week_9.dyna_ppo import DynaPPOAgent, set_seed

    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = DynaPPOAgent(
        env,
        use_model=True,
        **PPO_COMMON,
        **DYNA_SPECIFIC,
        seed=seed,
    )

    eval_env = gym.make("CartPole-v1")
    records = {"step": [], "one_step_mse": [], "reward_mse": [], "return": []}
    e_k_early = None
    e_k_late = None

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
                # Evaluate model
                metrics = agent.evaluate_model(num_samples=1000)
                records["step"].append(real_steps)
                records["one_step_mse"].append(metrics["state_mse"])
                records["reward_mse"].append(metrics["reward_mse"])

                # Evaluate policy
                mean_r, _ = agent.evaluate(eval_env, num_episodes=5)
                records["return"].append(mean_r)

                # Multi-step error at specific checkpoints
                if real_steps == 4000:
                    e_k_early = compute_multistep_error(
                        agent.model, agent.real_buffer, MAX_K
                    )
                if real_steps >= TOTAL_STEPS - EVAL_INTERVAL:
                    e_k_late = compute_multistep_error(
                        agent.model, agent.real_buffer, MAX_K
                    )

                print(
                    f"  Step {real_steps:6d} | MSE: {metrics['state_mse']:.4f} | Return: {mean_r:.1f}"
                )

        agent.update(trajectory)
        agent.store_real(trajectory)
        agent.train_model()
        agent.imagine_and_update()

    env.close()
    eval_env.close()
    return records, e_k_early, e_k_late


def main() -> None:
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_e_k_early = []
    all_e_k_late = []

    for seed in range(NUM_SEEDS):
        print(f"\nSeed {seed + 1}/{NUM_SEEDS}")
        records, e_k_early, e_k_late = run_experiment_with_model_eval(seed)
        all_records.append(records)
        if e_k_early is not None:
            all_e_k_early.append(e_k_early)
        if e_k_late is not None:
            all_e_k_late.append(e_k_late)

    # Aggregate
    steps = np.array(all_records[0]["step"])
    mse_mean = np.mean([r["one_step_mse"] for r in all_records], axis=0)
    mse_std = np.std(
        [r["one_step_mse"] for r in all_records], axis=0, ddof=0
    ) / np.sqrt(NUM_SEEDS)
    ret_mean = np.mean([r["return"] for r in all_records], axis=0)

    # Plot 1: One-step MSE vs real steps (with return on secondary axis)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        steps, mse_mean, "o-", color="#d1495b", linewidth=2, label="One-step state MSE"
    )
    ax1.fill_between(
        steps, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2, color="#d1495b"
    )
    ax1.set_xlabel("Real Environment Steps")
    ax1.set_ylabel("One-step MSE", color="#d1495b")
    ax1.tick_params(axis="y", labelcolor="#d1495b")

    ax2 = ax1.twinx()
    ax2.plot(steps, ret_mean, "s-", color="#1f77b4", linewidth=2, label="Avg Return")
    ax2.set_ylabel("Average Return", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    fig.suptitle("Dyna-PPO: Model Accuracy vs Performance", fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_mse_vs_steps.png", dpi=150)
    plt.close(fig)

    # Plot 2: Multi-step error E_k curves at early and late stages
    fig, ax = plt.subplots(figsize=(10, 6))
    k_range = np.arange(1, MAX_K + 1)

    if all_e_k_early:
        e_k_early_mean = np.mean(all_e_k_early, axis=0)
        ax.plot(
            k_range,
            e_k_early_mean,
            "o--",
            color="#d1495b",
            linewidth=2,
            label="Early (4k steps)",
        )

    if all_e_k_late:
        e_k_late_mean = np.mean(all_e_k_late, axis=0)
        ax.plot(
            k_range,
            e_k_late_mean,
            "s-",
            color="#2a9d8f",
            linewidth=2,
            label=f"Late ({TOTAL_STEPS} steps)",
        )

    ax.set_xlabel("Imagination Horizon k")
    ax.set_ylabel("E_k (MSE after k steps)")
    ax.set_title("Multi-step Prediction Error: E_k Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "multistep_error.png", dpi=150)
    plt.close(fig)

    # Summary
    print(f"\n{'=' * 50}")
    print("MODEL PREDICTION ACCURACY SUMMARY")
    print(f"{'=' * 50}")
    print(f"One-step MSE at 4k steps:  {mse_mean[1]:.4f}")
    print(f"One-step MSE at {TOTAL_STEPS} steps: {mse_mean[-1]:.4f}")
    if all_e_k_early and all_e_k_late:
        print(f"E_k at k=5 (early): {np.mean(all_e_k_early, axis=0)[4]:.4f}")
        print(f"E_k at k=5 (late):  {np.mean(all_e_k_late, axis=0)[4]:.4f}")
        print(f"E_k at k=10 (early): {np.mean(all_e_k_early, axis=0)[9]:.4f}")
        print(f"E_k at k=10 (late):  {np.mean(all_e_k_late, axis=0)[9]:.4f}")

    # Find MSE threshold where return starts improving
    ret_increases = np.diff(ret_mean) > 0
    if np.any(ret_increases):
        threshold_idx = np.argmax(ret_increases)
        print(
            f"\nReturn begins improving around MSE={mse_mean[threshold_idx]:.4f} (step {steps[threshold_idx]})"
        )

    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
