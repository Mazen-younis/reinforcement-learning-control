# =========================================================
# DC Motor Speed Control with SARSA Reinforcement Learning
# Single-file submission version (main.py)
# Mazen Younis & Mert Satici
#
#
# Run:
#   python main.py evaluate_policy
#   python main.py train_many
# =========================================================

"""
Single-file implementation of the DC motor RL controller.

Contains:
- DC motor environment (physics simulation + discretization for tabular RL)
- SARSA(0) agent (tabular temporal-difference control, on-policy)
- Antagonist (load-mode selector / disturbance manager)
- Training: "train_many"
- Evaluation: "evaluate_policy"

Run modes (one file, two scripts):
    python main.py train_many
    python main.py evaluate_policy

Notes:
- This file is self-contained: no project-relative imports.
- Uses only numpy + matplotlib from typical course toolchains.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


# =========================================================
# Environment (DC motor physics + RL interface)
# =========================================================

class DCMotorEnv:
    """
    DC Motor Speed Control Environment (Tabular RL)

    Objective:
    - Track a constant angular speed reference omega_ref [rad/s]
    - Constraints: rise time <= 1 ms (to 90% ref), overshoot <= 3%
    - Robustness: one policy should work for no/nominal/high load

    Continuous-time motor model:
      Electrical:
        di/dt = (V - R*i - Ke*omega) / L
      Mechanical:
        domega/dt = (Kt*i - b*omega - tau_load_dir) / J

    Key RL design:
    - Discrete action space: fixed voltage levels
    - Discrete state space: bins for (omega, error, current i)
      Including current i is important because it captures torque demand under load.
    """

    def __init__(
            self,
            dt: float = 1e-5,
            R: float = 0.331,
            L: float = 0.103e-3,
            Ke: float = 27.3e-3,
            Kt: float = 27.3e-3,
            J: float = 7.28e-6,
            b: float = 5.2e-6,
            tau_load: float = 0.0,
            V_max: float = 24.0,
            omega_ref: float = 100.0,
            T_episode: float = 0.02,
    ):
        # Simulation config
        self.dt = float(dt)
        self.T_episode = float(T_episode)
        self.max_steps = int(self.T_episode / self.dt)
        self.step_count = 0

        # Motor params
        self.R = float(R)
        self.L = float(L)
        self.Ke = float(Ke)
        self.Kt = float(Kt)
        self.J = float(J)
        self.b = float(b)

        # Limits / setpoint
        self.V_max = float(V_max)
        self.omega_ref = float(omega_ref)
        self.overshoot_limit = 1.03 * self.omega_ref

        # Load scenarios (assignment robustness requirement)
        tau_nominal = 0.108  # [Nm] 108 mNm
        self.load_modes: Dict[str, float] = {
            "no_load": 0.0,
            "nominal": 0.30 * tau_nominal,
            "high": 0.70 * tau_nominal,
        }

        self.tau_load = float(tau_load)
        self.load_mode = "custom"

        # Antagonist load schedule: list[(step_idx, mode)]
        self.load_schedule: Optional[List[Tuple[int, str]]] = None
        self._next_load_event_idx = 0

        # Discrete actions (voltage)
        self.actions = np.array([-24, -18, -12, -6, 0, 6, 12, 18, 24], dtype=np.float64)

        # Reward weights
        self.w_err = 3.0
        self.w_u = 0.01
        self.w_du = 0.25
        self.w_overshoot = 25.0
        self.w_prog = 0.15
        self.success_bonus = 2.0
        self.w_wrong = 0.75

        # Settling detection
        self.err_band = 0.02
        self.settle_steps = 50
        self._settle_counter = 0

        # Continuous state
        self.t = 0.0
        self.omega = 0.0
        self.i = 0.0
        self.prev_V = 0.0
        self.prev_err = 0.0

        # Discretization config (must match agent)
        self.n_omega_bins = 25
        self.n_err_bins = 25
        self.n_i_bins = 15

        # Current bounds (for discretization + numerical safety)
        self.i_min = -80.0
        self.i_max = 80.0
        self.i_hard_clip = 120.0

    def reset(
            self,
            omega0: float = 0.0,
            i0: float = 0.0,
            tau_load: Optional[float] = None,
            load_mode: Optional[str] = None,
            load_schedule: Optional[List[Tuple[int, str]]] = None,
            omega_ref: Optional[float] = None,
    ) -> np.ndarray:
        """Reset episode. Returns continuous state vector [omega, i, error]."""
        self.t = 0.0
        self.step_count = 0
        self._settle_counter = 0

        self.omega = float(omega0)
        self.i = float(i0)
        self.prev_V = 0.0

        if omega_ref is not None:
            self.omega_ref = float(omega_ref)
        self.overshoot_limit = 1.03 * self.omega_ref

        # Reset schedule
        self.load_schedule = None
        self._next_load_event_idx = 0
        self.load_mode = "custom"

        # Schedule (antagonist switching)
        if load_schedule is not None:
            self.load_schedule = list(load_schedule)
            self.load_schedule.sort(key=lambda x: int(x[0]))

            for k, (step_idx, mode) in enumerate(self.load_schedule):
                if int(step_idx) < 0:
                    raise ValueError(f"load_schedule[{k}] has negative step_idx={step_idx}.")
                if mode not in self.load_modes:
                    raise ValueError(
                        f"load_schedule[{k}] has unknown mode='{mode}'. "
                        f"Use one of {list(self.load_modes.keys())}."
                    )

            if len(self.load_schedule) > 0 and int(self.load_schedule[0][0]) == 0:
                _, m0 = self.load_schedule[0]
                self.tau_load = float(self.load_modes[m0])
                self.load_mode = str(m0)
                self._next_load_event_idx = 1

        # Fixed load (if no schedule)
        if self.load_schedule is None:
            if load_mode is not None:
                if load_mode not in self.load_modes:
                    raise ValueError(f"Unknown load_mode={load_mode}.")
                self.tau_load = float(self.load_modes[load_mode])
                self.load_mode = str(load_mode)
            elif tau_load is None:
                # label as known mode if it matches exactly
                for name, val in self.load_modes.items():
                    if abs(self.tau_load - float(val)) < 1e-12:
                        self.load_mode = name
                        break

        # Manual override always wins
        if tau_load is not None:
            self.tau_load = float(tau_load)
            self.load_mode = "custom"

        err = self.omega_ref - self.omega
        self.prev_err = err
        return np.array([self.omega, self.i, err], dtype=np.float64)

    def _directional_load(self, V: float, omega: float) -> float:
        """Signed resisting torque that opposes motion direction."""
        omega_deadband = 1e-2
        if abs(omega) <= omega_deadband:
            direction = np.sign(V) if abs(V) > 1e-12 else 0.0
        else:
            direction = np.sign(omega)
        return float(self.tau_load * direction)

    def _apply_load_schedule_if_needed(self) -> None:
        """Apply next load_schedule event if step_count matches its step index."""
        if self.load_schedule is None:
            return
        if self._next_load_event_idx >= len(self.load_schedule):
            return
        step_idx, mode = self.load_schedule[self._next_load_event_idx]
        if self.step_count == int(step_idx):
            self.tau_load = float(self.load_modes[mode])
            self.load_mode = str(mode)
            self._next_load_event_idx += 1

    def step(self, action_voltage: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Advance simulation by one dt step.

        Returns:
            state: [omega, i, error] (continuous)
            reward: shaped reward
            terminated: episode ended (time horizon or early-settle)
            info: diagnostics for plots/metrics
        """
        V_cmd = float(action_voltage)
        V = float(np.clip(V_cmd, -self.V_max, self.V_max))

        # antagonist schedule update
        self._apply_load_schedule_if_needed()
        tau_dir = self._directional_load(V, self.omega)

        # dynamics
        di_dt = (V - self.R * self.i - self.Ke * self.omega) / self.L
        domega_dt = (self.Kt * self.i - self.b * self.omega - tau_dir) / self.J

        # Euler integration
        i_next = float(self.i + self.dt * di_dt)
        i_next = float(np.clip(i_next, -self.i_hard_clip, self.i_hard_clip))
        self.i = i_next

        omega_next = float(self.omega + self.dt * domega_dt)
        if self.omega <= 0.0 and omega_next < 0.0:
            omega_next = 0.0
        self.omega = omega_next

        self.t += self.dt
        self.step_count += 1
        terminated = self.step_count >= self.max_steps

        # tracking error
        err = self.omega_ref - self.omega

        # reward shaping (normalized terms)
        e_norm = err / max(self.omega_ref, 1e-9)
        u_norm = V / max(self.V_max, 1e-9)
        du_norm = (V - self.prev_V) / max(self.V_max, 1e-9)

        reward = -self.w_err * (e_norm ** 2)
        reward -= self.w_u * (u_norm ** 2)
        reward -= self.w_du * (du_norm ** 2)
        reward += self.w_prog * (self.omega / max(self.omega_ref, 1e-9))

        # wrong-direction penalty
        if err > 0.0 and V < 0.0:
            reward -= self.w_wrong * abs(u_norm)
        elif err < 0.0 and V > 0.0:
            reward -= self.w_wrong * abs(u_norm)

        # overshoot penalty
        if self.omega > self.overshoot_limit:
            reward -= self.w_overshoot
            self._settle_counter = 0

        # settle detection
        in_band = (abs(err) <= self.err_band * self.omega_ref) and (self.omega <= self.overshoot_limit)
        if in_band:
            self._settle_counter += 1
            reward += self.success_bonus
            if self._settle_counter >= self.settle_steps:
                terminated = True
                reward += 10.0
        else:
            self._settle_counter = 0

        self.prev_V = V
        self.prev_err = err

        info = {
            "t": self.t,
            "step": self.step_count,
            "V_cmd": V_cmd,
            "V": V,
            "omega": self.omega,
            "i": self.i,
            "error": err,
            "tau_load_mag": self.tau_load,
            "tau_load_applied": tau_dir,
            "load_mode": self.load_mode,
            "settle_counter": self._settle_counter,
        }

        state = np.array([self.omega, self.i, err], dtype=np.float64)
        return state, float(reward), bool(terminated), info

    @staticmethod
    def _discretize(x: float, x_min: float, x_max: float, n_bins: int) -> int:
        """Linear binning with clipping."""
        if x_max <= x_min:
            raise ValueError(f"Invalid range: x_min={x_min}, x_max={x_max}")
        x = float(np.clip(x, x_min, x_max))
        return int((x - x_min) / (x_max - x_min) * (n_bins - 1))

    def get_discrete_state(self) -> Tuple[int, int, int]:
        """
        Discretize (omega, error, i) for tabular RL.

        Returns:
            (omega_bin, err_bin, i_bin)
        """
        err = self.omega_ref - self.omega

        omega_bin = self._discretize(self.omega, 0.0, 1.5 * self.omega_ref, self.n_omega_bins)
        err_bin = self._discretize(err, -self.omega_ref, self.omega_ref, self.n_err_bins)
        i_bin = self._discretize(self.i, self.i_min, self.i_max, self.n_i_bins)

        return omega_bin, err_bin, i_bin


# =========================================================
# SARSA(0) Agent (tabular)
# =========================================================

class SarsaAgent:
    """Tabular SARSA(0) with epsilon-greedy action selection."""

    def __init__(
            self,
            alpha: float = 0.2,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            n_omega_bins: int = 25,
            n_err_bins: int = 25,
            n_i_bins: int = 15,
            n_actions: int = 9,
            eps_min: float = 0.05,
            eps_decay: float = 0.995,
            seed: int = 0,
    ):
        self.n_omega_bins = int(n_omega_bins)
        self.n_err_bins = int(n_err_bins)
        self.n_i_bins = int(n_i_bins)
        self.n_actions = int(n_actions)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.eps_min = float(eps_min)
        self.eps_decay = float(eps_decay)

        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros(
            (self.n_omega_bins, self.n_err_bins, self.n_i_bins, self.n_actions),
            dtype=np.float64,
        )

    def select_action_index(self, state: Tuple[int, int, int], training: bool = True) -> int:
        """Epsilon-greedy with random tie-breaking for argmax."""
        o, e, i = state
        if training and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(self.n_actions))

        q = self.Q[o, e, i, :]
        max_q = float(np.max(q))
        best = np.flatnonzero(q == max_q)
        return int(self.rng.choice(best))

    def update(
            self,
            s: Tuple[int, int, int],
            a_idx: int,
            r: float,
            s_next: Tuple[int, int, int],
            a_next_idx: int,
            terminated: bool,
    ) -> None:
        """SARSA TD(0): Q(s,a) <- Q(s,a) + alpha*(target - Q(s,a))."""
        o1, e1, i1 = s
        q_sa = self.Q[o1, e1, i1, a_idx]

        if terminated:
            target = float(r)
        else:
            o2, e2, i2 = s_next
            target = float(r) + self.gamma * self.Q[o2, e2, i2, a_next_idx]

        self.Q[o1, e1, i1, a_idx] = q_sa + self.alpha * (target - q_sa)

    def end_episode(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# =========================================================
# Antagonist (load-mode selector / disturbance manager)
# =========================================================

class AntagonistAgent:
    """
    Antagonist for "SARSA with Antagonist".

    It does not control motor voltage. It selects which load mode to apply each episode.
    We track an EMA of returns per load mode and select the currently "hardest" load
    (lowest EMA return), with some exploration.
    """

    def __init__(
            self,
            load_modes: List[str],
            seed: int = 0,
            explore: float = 0.1,
            ema_alpha: float = 0.1,
    ):
        self.load_modes = list(load_modes)
        self.rng = np.random.default_rng(seed)
        self.explore = float(explore)
        self.ema_alpha = float(ema_alpha)

        self.counts: Dict[str, int] = {lm: 0 for lm in self.load_modes}
        self.means: Dict[str, float] = {lm: 0.0 for lm in self.load_modes}

    def select_load_mode(self) -> str:
        if self.rng.random() < self.explore:
            return str(self.rng.choice(self.load_modes))
        return min(self.load_modes, key=lambda lm: self.means[lm])

    def update(self, load_mode: str, episode_return: float) -> None:
        self.counts[load_mode] += 1
        if self.counts[load_mode] == 1:
            self.means[load_mode] = float(episode_return)
            return
        old = self.means[load_mode]
        a = self.ema_alpha
        self.means[load_mode] = float((1.0 - a) * old + a * float(episode_return))


# =========================================================
# Shared rollout (used by both training & evaluation)
# =========================================================

def rollout_episode(
        env: DCMotorEnv,
        agent: SarsaAgent,
        load_mode: str = "nominal",
        load_schedule: Optional[List[Tuple[int, str]]] = None,
        training: bool = True,
        min_duration_s: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one episode with optional minimum duration override.

    Returns:
        total_return, t_hist, omega_hist, v_hist, tau_hist
    """
    env.reset(load_mode=load_mode, load_schedule=load_schedule)

    t_hist: List[float] = []
    omega_hist: List[float] = []
    v_hist: List[float] = []
    tau_hist: List[float] = []

    s = env.get_discrete_state()
    a_idx = agent.select_action_index(s, training=training)

    total = 0.0
    while True:
        V_cmd = float(env.actions[a_idx])
        _, r, terminated, info = env.step(V_cmd)
        total += float(r)

        t_hist.append(float(info["t"]))
        omega_hist.append(float(info["omega"]))
        v_hist.append(float(info["V"]))
        tau_hist.append(float(info["tau_load_applied"]))

        s_next = env.get_discrete_state()
        a_next_idx = agent.select_action_index(s_next, training=training)

        if training:
            agent.update(s, a_idx, r, s_next, a_next_idx, terminated)

        s, a_idx = s_next, a_next_idx

        if terminated:
            if (min_duration_s is None) or (float(info["t"]) >= float(min_duration_s)):
                break

    return (
        float(total),
        np.array(t_hist, dtype=np.float64),
        np.array(omega_hist, dtype=np.float64),
        np.array(v_hist, dtype=np.float64),
        np.array(tau_hist, dtype=np.float64),
    )


# =========================================================
# train_many (learning curves)
# =========================================================

def exponential_moving_average(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    ema = np.empty_like(x)
    ema[0] = x[0]
    for k in range(1, len(x)):
        ema[k] = alpha * x[k] + (1.0 - alpha) * ema[k - 1]
    return ema


def plot_learning_curve(
        returns: List[float],
        eps_hist: List[float],
        title: str,
        save_path: Optional[Path] = None,
) -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    r = np.asarray(returns, dtype=np.float64)
    e = np.asarray(eps_hist, dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(r, alpha=0.25, linewidth=0.8, label="Return (raw)")
    ax1.plot(exponential_moving_average(r, alpha=0.05), linewidth=2.0, label="Return (EMA)")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return (sum of rewards)")
    ax1.set_title(title)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax1.ticklabel_format(axis="y", style="plain")

    ax2 = ax1.twinx()
    ax2.plot(e, linestyle="--", alpha=0.8, linewidth=1.2, label="Epsilon ε")
    ax2.set_ylabel("Exploration rate ε")
    ax2.set_ylim(0.0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


def train_many(
        n_episodes: int = 600,
        seed: int = 0,
        mode: str = "separate",
        out_dir: Path = Path("pictures"),
) -> None:
    """
    Training driver.

    mode="separate":
        Train one agent per load mode (3 curves). Good diagnostic.
    mode="robust":
        Train one agent with randomized loads (one curve). Good robust policy.
    """
    env = DCMotorEnv()
    load_modes = ["no_load", "nominal", "high"]

    if mode == "separate":
        for idx, lm in enumerate(load_modes):
            agent = SarsaAgent(seed=seed + idx)

            returns: List[float] = []
            eps_hist: List[float] = []

            for ep in range(n_episodes):
                eps_hist.append(agent.epsilon)
                G, *_ = rollout_episode(env, agent, load_mode=lm, load_schedule=None, training=True)
                returns.append(G)
                agent.end_episode()

                if (ep + 1) % 50 == 0:
                    print(f"[{lm}] Episode {ep + 1}/{n_episodes}: return={G:.2e}, eps={agent.epsilon:.3f}")

            plot_learning_curve(
                returns=returns,
                eps_hist=eps_hist,
                title=f"SARSA Training Curve ({lm.replace('_', ' ').title()})",
                save_path=(out_dir / f"training_curve_{lm}.png"),
            )
        return

    if mode == "robust":
        agent = SarsaAgent(seed=seed)
        rng = np.random.default_rng(seed)

        returns = []
        eps_hist = []
        for ep in range(n_episodes):
            eps_hist.append(agent.epsilon)
            lm = str(rng.choice(load_modes))
            G, *_ = rollout_episode(env, agent, load_mode=lm, load_schedule=None, training=True)
            returns.append(G)
            agent.end_episode()

            if (ep + 1) % 50 == 0:
                print(f"[robust] Episode {ep + 1}/{n_episodes}: load={lm}, return={G:.2e}, eps={agent.epsilon:.3f}")

        plot_learning_curve(
            returns=returns,
            eps_hist=eps_hist,
            title="SARSA Training Curve (Robust Agent, Randomized Loads)",
            save_path=(out_dir / "training_curve_robust.png"),
        )
        return

    raise ValueError("mode must be 'separate' or 'robust'.")


# =========================================================
# evaluate_policy (plots + metrics)
# =========================================================

EVAL_MIN_DURATION_S = 0.002
ANTAGONIST_MIN_DURATION_S = 0.003
EVAL_XLIM_MS = 2.0
ANTAGONIST_XLIM_MS = 3.0


def set_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "lines.linewidth": 1.6,
    })


def _to_ms(t_s: np.ndarray) -> np.ndarray:
    return 1e3 * np.asarray(t_s, dtype=np.float64)


def _clip_by_time(t: np.ndarray, y: np.ndarray, t_end_s: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if t_end_s is None:
        return t, y
    m = t <= float(t_end_s)
    return t[m], y[m]


def _downsample_for_plot(t: np.ndarray, y: np.ndarray, max_points: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n <= max_points:
        return t, y
    stride = max(1, n // max_points)
    return t[::stride], y[::stride]


def _moving_average(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(y, k, mode="same")


def compute_metrics(t: np.ndarray, omega: np.ndarray, omega_ref: float) -> Tuple[Optional[float], float, float]:
    target_90 = 0.90 * omega_ref
    idx = np.where(omega >= target_90)[0]
    rise_time_90 = float(t[idx[0]]) if len(idx) > 0 else None

    omega_max = float(np.max(omega))
    overshoot_pct = max(0.0, (omega_max - omega_ref) / omega_ref * 100.0)
    steady_state_error = float(omega_ref - omega[-1])
    return rise_time_90, overshoot_pct, steady_state_error


def print_voltage_stats(t: np.ndarray, omega: np.ndarray, v: np.ndarray, omega_ref: float, window_s: float = 0.005) -> None:
    if len(t) < 2:
        print("\n--- Voltage stats: not enough samples ---")
        return

    t_end = float(t[-1])
    actual_window = min(float(window_s), max(0.0, t_end - float(t[0])))
    t_start = max(float(t[0]), t_end - actual_window)
    mask = t >= t_start

    t_w, omega_w, v_w = t[mask], omega[mask], v[mask]
    if len(t_w) < 2:
        print("\n--- Voltage stats: window too small ---")
        return

    uniq, counts = np.unique(v_w, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    total = int(np.sum(counts))

    switches = int(np.sum(v_w[1:] != v_w[:-1]))
    duration = float(t_w[-1] - t_w[0])
    switch_rate = (switches / duration) if duration > 0 else 0.0

    err_w = omega_ref - omega_w
    wrong = np.logical_or((err_w > 0) & (v_w < 0), (err_w < 0) & (v_w > 0))
    wrong_frac = float(np.mean(wrong)) if len(wrong) > 0 else 0.0

    print(f"\n--- Voltage stats over last {window_s * 1e3:.1f} ms (t in [{t_start:.6f}, {t_end:.6f}]) ---")
    print(f"Switches: {switches} over {duration * 1e3:.3f} ms  ->  {switch_rate:.1f} switches/s")
    print(f"Wrong-direction fraction: {wrong_frac * 100:.2f}%")
    print("Top voltages used (value: percent):")
    for u, c in zip(uniq[:10], counts[:10]):
        print(f"  {u:>6.1f} V : {100.0 * c / total:6.2f}%")


def plot_speed_rollouts_all_loads(env: DCMotorEnv, rollouts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                                  title: str, xlim_ms: float, save_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(9, 5.2))
    ax = plt.gca()

    for lm, (t, omega, v, tau) in rollouts.items():
        t_c, o_c = _clip_by_time(t, omega, t_end_s=xlim_ms * 1e-3)
        t_ds, o_ds = _downsample_for_plot(t_c, o_c, max_points=2000)
        ax.plot(_to_ms(t_ds), o_ds, label=lm)

    ax.axhline(env.omega_ref, linestyle="--", color="k", label="omega_ref")
    ax.axhline(env.overshoot_limit, linestyle=":", color="k", label="3% overshoot limit")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Angular speed ω [rad/s]")
    ax.set_title(title)
    ax.set_xlim(0.0, float(xlim_ms))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


def plot_speed_rollout_single(env: DCMotorEnv, t: np.ndarray, omega: np.ndarray, lm: str,
                              xlim_ms: float, save_path: Optional[Path] = None) -> None:
    t_c, o_c = _clip_by_time(t, omega, t_end_s=xlim_ms * 1e-3)
    t_ds, o_ds = _downsample_for_plot(t_c, o_c, max_points=2000)

    plt.figure(figsize=(9, 5.2))
    plt.plot(_to_ms(t_ds), o_ds, label=lm)

    plt.axhline(env.omega_ref, linestyle="--", color="k", label="omega_ref")
    plt.axhline(env.overshoot_limit, linestyle=":", color="k", label="3% overshoot limit")

    plt.xlabel("Time [ms]")
    plt.ylabel("Angular speed ω [rad/s]")
    plt.title(f"Greedy Policy Speed Rollout ({lm})")
    plt.xlim(0.0, float(xlim_ms))
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


def plot_voltage_paper_quality(env: DCMotorEnv, t: np.ndarray, v: np.ndarray, title: str,
                               xlim_ms: float, save_path: Optional[Path] = None) -> None:
    t_c, v_c = _clip_by_time(t, v, t_end_s=xlim_ms * 1e-3)
    t_ds, v_ds = _downsample_for_plot(t_c, v_c, max_points=1500)
    v_env = _moving_average(v_c, w=50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)

    ax1.step(_to_ms(t_ds), v_ds, where="post")
    ax1.set_ylabel("Voltage V [V]")
    ax1.set_title(title)
    ax1.set_yticks(env.actions)
    ax1.set_ylim(float(np.min(env.actions)) - 2.0, float(np.max(env.actions)) + 2.0)

    ax2.plot(_to_ms(t_c), v_env)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("V (moving avg) [V]")
    ax2.set_xlim(0.0, float(xlim_ms))

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


def plot_voltage_histogram(env: DCMotorEnv, v: np.ndarray, title: str, save_path: Optional[Path] = None) -> None:
    uniq, counts = np.unique(v, return_counts=True)
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]

    plt.figure(figsize=(9, 4.6))
    plt.bar(uniq, counts / np.sum(counts) * 100.0, width=3.5)
    plt.xticks(env.actions)
    plt.xlabel("Voltage level [V]")
    plt.ylabel("Usage [%]")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


def plot_antagonist_switching(env: DCMotorEnv, t: np.ndarray, omega: np.ndarray, tau: np.ndarray,
                              title: str, xlim_ms: float, save_path: Optional[Path] = None) -> None:
    t1, o1 = _clip_by_time(t, omega, t_end_s=xlim_ms * 1e-3)
    t2, tau2 = _clip_by_time(t, tau, t_end_s=xlim_ms * 1e-3)

    t_ds1, o_ds = _downsample_for_plot(t1, o1, max_points=2500)
    t_ds2, tau_ds = _downsample_for_plot(t2, tau2, max_points=2500)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)

    ax1.plot(_to_ms(t_ds1), o_ds, label="ω(t)")
    ax1.axhline(env.omega_ref, linestyle="--", color="k", label="omega_ref")
    ax1.axhline(env.overshoot_limit, linestyle=":", color="k", label="3% overshoot limit")
    ax1.set_ylabel("ω [rad/s]")
    ax1.set_title(title)
    ax1.legend()

    ax2.step(_to_ms(t_ds2), tau_ds, where="post", label="Applied load torque τ_load (signed)")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("τ_load [Nm]")
    ax2.legend()
    ax2.set_xlim(0.0, float(xlim_ms))

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


def train_agent_with_antagonist(
        env: DCMotorEnv,
        n_episodes: int = 800,
        training_loads: Tuple[str, ...] = ("no_load", "nominal", "high"),
        seed: int = 0,
        antagonist_explore: float = 0.10,
        antagonist_ema_alpha: float = 0.10,
        warmup_episodes: int = 120,
        mix_random_prob: float = 0.30,
) -> Tuple[SarsaAgent, AntagonistAgent]:
    agent = SarsaAgent(seed=seed)
    loads = list(training_loads)
    rng = np.random.default_rng(seed)

    antagonist = AntagonistAgent(
        load_modes=loads,
        seed=seed,
        explore=antagonist_explore,
        ema_alpha=antagonist_ema_alpha,
    )

    for ep in range(n_episodes):
        if ep < warmup_episodes:
            lm = loads[ep % len(loads)]
        else:
            if rng.random() < float(mix_random_prob):
                lm = str(rng.choice(loads))
            else:
                lm = antagonist.select_load_mode()

        G, *_ = rollout_episode(env, agent, load_mode=lm, load_schedule=None, training=True, min_duration_s=None)
        antagonist.update(load_mode=lm, episode_return=float(G))
        agent.end_episode()

    return agent, antagonist


def evaluate_policy(
        n_train: int = 1000,
        seed: int = 0,
        save_figures: bool = True,
        out_dir: Path = Path("pictures"),
) -> None:
    set_paper_style()

    env = DCMotorEnv()

    agent, antagonist = train_agent_with_antagonist(
        env,
        n_episodes=int(n_train),
        training_loads=("no_load", "nominal", "high"),
        seed=int(seed),
        antagonist_explore=0.10,
        antagonist_ema_alpha=0.10,
        warmup_episodes=120,
        mix_random_prob=0.30,
    )

    print("\n=== Antagonist summary (EMA return per load, lower = harder) ===")
    for lm in ["no_load", "nominal", "high"]:
        print(f"{lm:8s} : ema_return = {antagonist.means[lm]: .3e} (n={antagonist.counts[lm]})")

    agent.epsilon = 0.0  # greedy eval

    load_modes = ["no_load", "nominal", "high"]
    results: Dict[str, Tuple[float, Optional[float], float, float]] = {}
    rollouts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # fixed-load rollouts (force minimum duration for consistent plots)
    for lm in load_modes:
        G, t, omega, v, tau = rollout_episode(
            env, agent,
            load_mode=lm,
            load_schedule=None,
            training=False,
            min_duration_s=EVAL_MIN_DURATION_S,
        )
        rise_t, os_pct, ss_err = compute_metrics(t, omega, env.omega_ref)
        results[lm] = (G, rise_t, os_pct, ss_err)
        rollouts[lm] = (t, omega, v, tau)

    plot_speed_rollouts_all_loads(
        env=env,
        rollouts=rollouts,
        title="Greedy Policy Speed Rollouts (All Loads)",
        xlim_ms=EVAL_XLIM_MS,
        save_path=(out_dir / "eval_speed_all_loads.png") if save_figures else None,
    )

    for lm in load_modes:
        t, omega, v, tau = rollouts[lm]
        plot_speed_rollout_single(
            env=env,
            t=t,
            omega=omega,
            lm=lm,
            xlim_ms=EVAL_XLIM_MS,
            save_path=(out_dir / f"eval_speed_{lm}.png") if save_figures else None,
        )

    for lm in load_modes:
        t, omega, v, tau = rollouts[lm]
        print_voltage_stats(t, omega, v, env.omega_ref, window_s=0.005)

        plot_voltage_paper_quality(
            env=env,
            t=t,
            v=v,
            title=f"Control Signal ({lm}, Greedy Policy) — Step Command + Envelope",
            xlim_ms=EVAL_XLIM_MS,
            save_path=(out_dir / f"eval_voltage_{lm}.png") if save_figures else None,
        )

        plot_voltage_histogram(
            env=env,
            v=v,
            title=f"Voltage Usage Histogram ({lm}, Greedy Policy)",
            save_path=(out_dir / f"eval_voltage_hist_{lm}.png") if save_figures else None,
        )

    # antagonist switching stress test (visible torque steps)
    schedule = [(0, "no_load"), (100, "nominal"), (200, "high")]
    G, t, omega, v, tau = rollout_episode(
        env, agent,
        load_mode="no_load",
        load_schedule=schedule,
        training=False,
        min_duration_s=ANTAGONIST_MIN_DURATION_S,
    )

    plot_antagonist_switching(
        env=env,
        t=t,
        omega=omega,
        tau=tau,
        title="Greedy Policy with Antagonist Load Switching (Speed + Disturbance)",
        xlim_ms=ANTAGONIST_XLIM_MS,
        save_path=(out_dir / "eval_antagonist_switching_speed_tau.png") if save_figures else None,
    )

    plot_voltage_paper_quality(
        env=env,
        t=t,
        v=v,
        title="Control Signal with Antagonist Load Switching — Step Command + Envelope",
        xlim_ms=ANTAGONIST_XLIM_MS,
        save_path=(out_dir / "eval_antagonist_switching_voltage.png") if save_figures else None,
    )

    print("\n=== Quantitative Metrics (Greedy Policy) ===")
    print("load_mode | return | rise_time_90 [ms] | overshoot [%] | steady_state_error [rad/s]")
    for lm in load_modes:
        G, rise_t, os_pct, ss_err = results[lm]
        rise_ms = (rise_t * 1e3) if rise_t is not None else None
        print(f"{lm:8s} | {G: .3e} | {str(rise_ms):>15} | {os_pct:10.3f} | {ss_err: .3f}")


# =========================================================
# Single entry point (CLI)
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file SARSA DC motor project")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train_many", help="Run training curves (train_many)")
    p_train.add_argument("--episodes", type=int, default=600)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--mode", choices=["separate", "robust"], default="separate")
    p_train.add_argument("--out_dir", type=str, default="pictures")

    p_eval = sub.add_parser("evaluate_policy", help="Train + evaluate greedy policy (evaluate_policy)")
    p_eval.add_argument("--train_episodes", type=int, default=1000)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--no_save", action="store_true", help="Do not save figures")
    p_eval.add_argument("--out_dir", type=str, default="pictures")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.cmd == "train_many":
        train_many(
            n_episodes=int(args.episodes),
            seed=int(args.seed),
            mode=str(args.mode),
            out_dir=Path(args.out_dir),
        )
        return

    if args.cmd == "evaluate_policy":
        evaluate_policy(
            n_train=int(args.train_episodes),
            seed=int(args.seed),
            save_figures=not bool(args.no_save),
            out_dir=Path(args.out_dir),
        )
        return

    raise RuntimeError("Unknown command.")


if __name__ == "__main__":
    main()
