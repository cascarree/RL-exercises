"""
Generate all plots for Dyna-PPO Levels 1 & 2.
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
TOTAL_STEPS = 10000
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
    recs, model_mses = [], []
    rs = 0
    while rs < total_steps:
        state, _ = env.reset(seed=seed)
        done = False
        traj = []
        while not done and rs < total_steps:
            a, lp, e, v = agent.predict(state)
            ns, r, t, tr, _ = env.step(a)
            done = t or tr
            traj.append((state, a, lp, e, r, float(done), ns))
            state = ns
            rs += 1
            if rs % EVAL_INTERVAL == 0:
                mr, _ = agent.evaluate(eval_env, num_episodes=3)
                recs.append({"step": rs, "return": mr})
                mm = agent.evaluate_model(num_samples=500)
                model_mses.append({"step": rs, "mse": mm["state_mse"]})
        agent.update(traj)
        agent.store_real(traj)
        agent.train_model()
        agent.imagine_and_update()
    env.close()
    eval_env.close()
    return recs, model_mses, agent


def agg(results):
    steps = np.array([r["step"] for r in results[0]])
    rets = np.stack([np.array([r["return"] for r in res]) for res in results])
    m, s = rets.mean(0), rets.std(0, ddof=0) / np.sqrt(rets.shape[0])
    return steps, m, s


def plot_ci(ax, steps, mean, se, label, color):
    ax.plot(steps, mean, label=label, lw=2, color=color)
    ax.fill_between(steps, mean - se, mean + se, alpha=0.15, color=color)


# --- 1.1: Dyna-PPO vs PPO ---
def plot_1_1():
    print("1.1 Dyna-PPO vs PPO")
    ppo_res, dyna_res = [], []
    for s in range(NUM_SEEDS):
        r, _, _ = run_single(s, TOTAL_STEPS, imag_horizon=5, imag_batches=20)
        ppo_res.append(r)
        r2, _, _ = run_single(s + 100, TOTAL_STEPS, imag_horizon=5, imag_batches=20)
        dyna_res.append(r2)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res, c in [
        ("PPO", ppo_res, "#1f77b4"),
        ("Dyna-PPO", dyna_res, "#ff7f0e"),
    ]:
        s, m, se = agg(res)
        plot_ci(ax, s, m, se, name, c)
    ax.set_xlabel("Real Steps")
    ax.set_ylabel("Avg Return")
    ax.set_title("1.1: Dyna-PPO vs PPO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "dyna_ppo_vs_ppo.png", dpi=150)
    plt.close()


# --- 1.2: Model MSE vs steps + multi-step error ---
def plot_1_2():
    print("1.2 Model Accuracy")
    all_mses = []
    e_k_early, e_k_late = None, None
    for s in range(NUM_SEEDS):
        recs, mses, agent = run_single(
            s + 200, TOTAL_STEPS, imag_horizon=5, imag_batches=20
        )
        all_mses.append(mses)
        # Multi-step error
        if len(agent.real_buffer) > 500:
            e_k = compute_e_k(agent.model, agent.real_buffer, max_k=20)
            if s == 0:
                e_k_early = e_k
            else:
                e_k_late = e_k
    steps = np.array([r["step"] for r in all_mses[0]])
    mse_vals = np.mean([[r["mse"] for r in ms] for ms in all_mses], axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mse_vals, "o-", color="#d1495b", lw=2)
    ax.set_xlabel("Real Steps")
    ax.set_ylabel("One-step MSE")
    ax.set_title("1.2: Model One-step MSE vs Real Steps")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_mse_vs_steps.png", dpi=150)
    plt.close()
    # E_k plot
    fig, ax = plt.subplots(figsize=(10, 6))
    k = np.arange(1, 21)
    if e_k_early is not None:
        ax.plot(k, e_k_early, "o--", color="#d1495b", lw=2, label="Early")
    if e_k_late is not None:
        ax.plot(k, e_k_late, "s-", color="#2a9d8f", lw=2, label="Late")
    ax.set_xlabel("k")
    ax.set_ylabel("E_k (MSE after k steps)")
    ax.set_title("1.2: Multi-step Error E_k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "multistep_error.png", dpi=150)
    plt.close()


def compute_e_k(model, buf, max_k=20, n=100):
    errors = []
    for _ in range(n):
        s, a, r, s2, _ = buf[np.random.randint(len(buf))]
        state = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        e_k = []
        cur = state.clone()
        for k in range(max_k):
            a_oh = torch.zeros(1, model.fc1.in_features - cur.shape[1])
            a_oh[0, a] = 1.0
            with torch.no_grad():
                d, _ = model(cur, a_oh)
                nxt = cur + d
            e_k.append(
                float(
                    torch.nn.MSELoss()(
                        nxt, torch.tensor(s2, dtype=torch.float32).unsqueeze(0)
                    )
                )
            )
            cur = nxt.detach()
        errors.append(e_k)
    return np.mean(errors, axis=0)


# --- 2.1A: Horizon sweep ---
def plot_2_1a():
    print("2.1A Horizon")
    horizons = [1, 3, 5, 10]
    colors = ["#d1495b", "#e8a838", "#2a9d8f", "#30638e"]
    finals = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for h, c in zip(horizons, colors):
        res = [
            run_single(s + h * 10, TOTAL_STEPS, imag_horizon=h)
            for s in range(NUM_SEEDS)
        ]
        recs = [r[0] for r in res]
        s, m, se = agg(recs)
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


# --- 2.1B: Budget regimes ---
def plot_2_1b():
    print("2.1B Budget")
    regimes = {
        "Conservative(e=1,b=5)": dict(model_epochs=1, imag_batches=5),
        "Balanced(e=3,b=10)": dict(model_epochs=3, imag_batches=10),
        "Aggressive(e=5,b=20)": dict(model_epochs=5, imag_batches=20),
    }
    colors = ["#2a9d8f", "#e8a838", "#d1495b"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, p), c in zip(regimes.items(), colors):
        res = [run_single(s + 500, TOTAL_STEPS, **p) for s in range(NUM_SEEDS)]
        recs = [r[0] for r in res]
        s, m, se = agg(recs)
        plot_ci(ax, s, m, se, name, c)
    ax.set_xlabel("Real Steps")
    ax.set_ylabel("Avg Return")
    ax.set_title("2.1B: Budget Regimes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "budget_regimes.png", dpi=150)
    plt.close()


# --- 2.1C: Buffer size ---
def plot_2_1c():
    print("2.1C Buffer")
    sizes = [1000, 5000, 10000]
    colors = ["#d1495b", "#2a9d8f", "#30638e"]
    finals = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for bs, c in zip(sizes, colors):
        res = [
            run_single(s + 600, TOTAL_STEPS, max_buffer_size=bs)
            for s in range(NUM_SEEDS)
        ]
        recs = [r[0] for r in res]
        s, m, se = agg(recs)
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


# --- 2.2: Distribution shift ---
def plot_2_2():
    print("2.2 Distribution Shift")
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
    checkpoints = {"early(2k)": 2000, "mid(5k)": 5000, "late(8k)": 8000}
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
        return float(torch.nn.MSELoss()(st + d, s2t))

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


# --- 2.3: Failure mode ---
def plot_2_3():
    print("2.3 Failure Mode")
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.2]
    finals = []
    for sigma in sigmas:
        res = [
            run_single(s + 700, TOTAL_STEPS, noisy=True, sigma=sigma)
            for s in range(NUM_SEEDS)
        ]
        recs = [r[0] for r in res]
        _, m, _ = agg(recs)
        finals.append(m[-3:].mean())
        print(f"  sigma={sigma:.2f}: {finals[-1]:.1f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, finals, "o-", lw=2, color="#d1495b", ms=8)
    ax.set_xlabel("Model Noise (sigma)")
    ax.set_ylabel("Final Return")
    ax.set_title("2.3: Performance vs Model Noise")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_mode.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_1_1()
    plot_1_2()
    plot_2_1a()
    plot_2_1b()
    plot_2_1c()
    plot_2_2()
    plot_2_3()
    print(f"\nAll plots saved to {PLOTS_DIR}")
