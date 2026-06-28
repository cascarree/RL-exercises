"""
Level 2: Advanced Analyses & Ablations for Dyna-PPO (lightweight version)
"""

from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
NUM_SEEDS = 2
TOTAL_STEPS = 8000
EVAL_INTERVAL = 500

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


def run_single(seed, total_steps, noisy=False, sigma=0.0, **dk):
    from rl_exercises.week_9.dyna_ppo import DynaPPOAgent, set_seed

    env = gym.make("CartPole-v1")
    set_seed(env, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = dict(
        model_lr=1e-3,
        model_epochs=5,
        model_batch_size=64,
        imag_horizon=5,
        imag_batches=20,
        max_buffer_size=100000,
    )
    d.update(dk)
    agent = DynaPPOAgent(env, use_model=True, **PPO_COMMON, **d, seed=seed)

    if noisy and sigma > 0:
        orig = agent.model.forward

        def corrupt(s, a):
            delta, r = orig(s, a)
            return delta + torch.randn_like(delta) * sigma, r + torch.randn_like(
                r
            ) * sigma

        agent.model.forward = corrupt

    eval_env = gym.make("CartPole-v1")
    recs = []
    rs = 0
    while rs < total_steps:
        state, _ = env.reset(seed=seed)
        done = False
        traj = []
        while not done and rs < total_steps:
            action, logp, ent, val = agent.predict(state)
            ns, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            traj.append((state, action, logp, ent, reward, float(done), ns))
            state = ns
            rs += 1
            if rs % EVAL_INTERVAL == 0:
                mr, _ = agent.evaluate(eval_env, num_episodes=3)
                recs.append({"step": rs, "return": mr})
        agent.update(traj)
        agent.store_real(traj)
        agent.train_model()
        agent.imagine_and_update()
    env.close()
    eval_env.close()
    return recs


def agg(results):
    steps = np.array([r["step"] for r in results[0]])
    rets = np.stack([np.array([r["return"] for r in res]) for res in results])
    m, s = rets.mean(0), rets.std(0, ddof=0) / np.sqrt(rets.shape[0])
    return steps, m, s


def plot_ci(ax, steps, mean, se, label, color):
    ax.plot(steps, mean, label=label, lw=2, color=color)
    ax.fill_between(steps, mean - se, mean + se, alpha=0.15, color=color)


# 2.1A: Horizon sweep
def exp_2_1a():
    print("=== 2.1A: Horizon ===")
    horizons = [1, 3, 5, 10]
    colors = ["#d1495b", "#e8a838", "#2a9d8f", "#30638e"]
    finals = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for h, c in zip(horizons, colors):
        res = [run_single(s, TOTAL_STEPS, imag_horizon=h) for s in range(NUM_SEEDS)]
        s, m, se = agg(res)
        plot_ci(ax, s, m, se, f"h={h}", c)
        finals.append(m[-3:].mean())
    ax.set_xlabel("Real Steps")
    ax.set_ylabel("Avg Return")
    ax.set_title("2.1A: Imagination Horizon Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "horizon_sweep.png", dpi=150)
    plt.close()
    best = horizons[np.argmax(finals)]
    print(f"  Best: h={best} (return={max(finals):.1f})")
    return horizons, finals


# 2.1B: Budget regimes
def exp_2_1b():
    print("=== 2.1B: Budget ===")
    regimes = {
        "Conservative (e=1,b=5)": dict(model_epochs=1, imag_batches=5),
        "Balanced (e=3,b=10)": dict(model_epochs=3, imag_batches=10),
        "Aggressive (e=5,b=20)": dict(model_epochs=5, imag_batches=20),
    }
    colors = ["#2a9d8f", "#e8a838", "#d1495b"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, p), c in zip(regimes.items(), colors):
        res = [run_single(s, TOTAL_STEPS, **p) for s in range(NUM_SEEDS)]
        s, m, se = agg(res)
        plot_ci(ax, s, m, se, name, c)
    ax.set_xlabel("Real Steps")
    ax.set_ylabel("Avg Return")
    ax.set_title("2.1B: Budget Regimes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "budget_regimes.png", dpi=150)
    plt.close()


# 2.1C: Buffer size
def exp_2_1c():
    print("=== 2.1C: Buffer ===")
    sizes = [1000, 5000, 10000]
    colors = ["#d1495b", "#2a9d8f", "#30638e"]
    finals = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for bs, c in zip(sizes, colors):
        res = [run_single(s, TOTAL_STEPS, max_buffer_size=bs) for s in range(NUM_SEEDS)]
        s, m, se = agg(res)
        plot_ci(ax, s, m, se, f"buf={bs}", c)
        finals.append(m[-3:].mean())
    ax.set_xlabel("Real Steps")
    ax.set_ylabel("Avg Return")
    ax.set_title("2.1C: Buffer Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "buffer_sweep.png", dpi=150)
    plt.close()
    best = sizes[np.argmax(finals)]
    print(f"  Best: buf={best} (return={max(finals):.1f})")
    return sizes, finals


# 2.2: Distribution shift
def exp_2_2():
    print("=== 2.2: Distribution Shift ===")
    import torch.nn as nn
    from rl_exercises.week_9.dyna_ppo import DynaPPOAgent, set_seed

    env = gym.make("CartPole-v1")
    set_seed(env, 0)
    torch.manual_seed(0)
    np.random.seed(0)
    agent = DynaPPOAgent(
        env,
        use_model=True,
        **PPO_COMMON,
        model_lr=1e-3,
        model_epochs=5,
        model_batch_size=64,
        imag_horizon=5,
        imag_batches=20,
        max_buffer_size=100000,
        seed=0,
    )

    checkpoints = {"early(2k)": 2000, "mid(5k)": 5000, "late(7k)": 7000}
    bufs = {}
    rs = 0
    while rs < TOTAL_STEPS:
        state, _ = env.reset(seed=0)
        done = False
        traj = []
        while not done and rs < TOTAL_STEPS:
            a, lp, e, v = agent.predict(state)
            ns, r, t, tr, _ = env.step(a)
            done = t or tr
            traj.append((state, a, lp, e, r, float(done), ns))
            state = ns
            rs += 1
            for nm, tgt in checkpoints.items():
                if rs == tgt:
                    bufs[nm] = list(agent.real_buffer)
        agent.update(traj)
        agent.store_real(traj)
        agent.train_model()
        agent.imagine_and_update()
    env.close()

    def model_err(buf):
        samp = random.sample(buf, min(300, len(buf)))
        s, a, r, s2, _ = zip(*samp)
        st = torch.tensor(np.array(s), dtype=torch.float32)
        at = torch.tensor(np.array(a), dtype=torch.long)
        oh = torch.zeros(len(at), agent.model.fc1.in_features - st.shape[1])
        oh.scatter_(1, at.unsqueeze(1), 1.0)
        s2t = torch.tensor(np.array(s2), dtype=torch.float32)
        with torch.no_grad():
            d, _ = agent.model(st, oh)
        return float(nn.MSELoss()(st + d, s2t))

    results = {}
    for nm, buf in bufs.items():
        mid = len(buf) // 2
        old_mse = model_err(buf[:mid]) if mid > 10 else 0
        new_mse = model_err(buf[mid:]) if mid > 10 else 0
        results[nm] = {"old": old_mse, "new": new_mse}
        print(f"  {nm}: old={old_mse:.6f}, new={new_mse:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    stages = list(results.keys())
    x = np.arange(len(stages))
    w = 0.35
    ax.bar(
        x - w / 2,
        [results[s]["old"] for s in stages],
        w,
        label="Old states",
        color="#1f77b4",
    )
    ax.bar(
        x + w / 2,
        [results[s]["new"] for s in stages],
        w,
        label="New states",
        color="#ff7f0e",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Model MSE")
    ax.set_title("2.2: Distribution Shift")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "distribution_shift.png", dpi=150)
    plt.close()


# 2.3: Failure mode
def exp_2_3():
    print("=== 2.3: Failure Mode ===")
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.2]
    finals = []
    for sigma in sigmas:
        res = [
            run_single(s, TOTAL_STEPS, noisy=True, sigma=sigma)
            for s in range(NUM_SEEDS)
        ]
        _, m, _ = agg(res)
        finals.append(m[-3:].mean())
        print(f"  sigma={sigma:.2f}: return={finals[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, finals, "o-", lw=2, color="#d1495b", ms=8)
    ax.set_xlabel("Model Noise (sigma)")
    ax.set_ylabel("Final Avg Return")
    ax.set_title("2.3: Performance vs Model Noise")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_mode.png", dpi=150)
    plt.close()
    return sigmas, finals


if __name__ == "__main__":
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_2_1a()
    exp_2_1b()
    exp_2_1c()
    exp_2_2()
    exp_2_3()
    print(f"\nAll plots saved to {PLOTS_DIR}")
