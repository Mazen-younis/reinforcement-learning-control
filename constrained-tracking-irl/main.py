# =========================================================
# Integral Reinforcement Learning (IRL) – Single-file submission
# Team 3: Mazen Younis & Mert Satici
#
# This file contains EVERYTHING needed to run the project:
# - Spring–mass–damper plant + sine reference
# - RK4 integrator
# - Tracking helpers (error + augmented state)
# - Constrained tracking cost (with saturated control cost)
# - Quadratic polynomial basis (features + Jacobian)
# - IRL rolling window buffer
# - IRL critic + actor
# - Plot/log utilities
# - Four runnable “scripts” via CLI:
#     1) run_pd
#     2) run_irl
#     3) run_blend
#     4) sweep_beta
#
# Run:
#   python main.py -h
#   python main.py run_pd
#   python main.py run_irl
#   python main.py run_blend
#   python main.py sweep_beta
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# util/utils.py
# =========================================================

def clip(u: np.ndarray, low: float, high: float) -> np.ndarray:
    """Elementwise clipping (safety guard on actuator)."""
    return np.minimum(np.maximum(u, low), high)


def rms(x: np.ndarray) -> float:
    """Root-mean-square of a 1D array."""
    x = np.asarray(x).reshape(-1)
    return float(np.sqrt(np.mean(x * x)))


# =========================================================
# sim/integrators.py
# =========================================================

DynamicsFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RK4:
    """
    Explicit RK4 integrator for xdot = f(t, x, u), with zero-order hold on u.
    """
    dt: float

    def step(self, f: DynamicsFn, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dt = self.dt
        k1 = f(t, x, u)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        k4 = f(t + dt, x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =========================================================
# sim/tracking.py
# =========================================================

def tracking_error(x: np.ndarray, x_d: np.ndarray) -> np.ndarray:
    """e_d = x - x_d."""
    return np.asarray(x, dtype=float) - np.asarray(x_d, dtype=float)


def augmented_state(x: np.ndarray, x_d: np.ndarray) -> np.ndarray:
    """
    X = [e_d; x_d].
    For spring–mass–damper: X = [e_pos, e_vel, x_d_pos, x_d_vel] (dimension 4).
    """
    e_d = tracking_error(x, x_d)
    return np.concatenate([e_d, np.asarray(x_d, dtype=float)], axis=0)


# =========================================================
# env/spring_mass_damper.py
# =========================================================

@dataclass
class SpringMassDamper:
    """
    Linear spring–mass–damper system.

    State:   x = [position, velocity]
    Control: u (scalar, provided as shape (1,))
    """
    m: float
    c: float
    k: float

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x1, x2 = x
        uu = float(u[0])
        x1_dot = x2
        x2_dot = -(self.k / self.m) * x1 - (self.c / self.m) * x2 + (1.0 / self.m) * uu
        return np.array([x1_dot, x2_dot], dtype=float)


# =========================================================
# env/reference.py
# =========================================================

@dataclass(frozen=True)
class SineReference:
    """
    Sinusoidal reference:
      x_d1(t) = A sin(wt)
      x_d2(t) = A w cos(wt)
    """
    amplitude: float
    omega: float

    def xd(self, t: float) -> np.ndarray:
        A, w = self.amplitude, self.omega
        return np.array([A * np.sin(w * t), A * w * np.cos(w * t)], dtype=float)

    def xd_dot(self, t: float) -> np.ndarray:
        A, w = self.amplitude, self.omega
        return np.array([A * w * np.cos(w * t), -A * (w ** 2) * np.sin(w * t)], dtype=float)


# =========================================================
# sim/costs.py
# =========================================================

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically safe log to avoid log(0)."""
    return np.log(np.maximum(x, eps))


@dataclass(frozen=True)
class ConstrainedTrackingCost:
    """
    r(X,u) = e^T Q e + U(u)
    where U is a non-quadratic cost yielding a saturated optimal policy.

    Assumption here (matches our scripts):
    - Scalar input (m=1)
    - R is diagonal (1x1)
    """
    Q: np.ndarray
    R: np.ndarray
    lam: float

    def __post_init__(self):
        Q = np.asarray(self.Q, dtype=float)
        R = np.asarray(self.R, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square matrix.")
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError("R must be a square matrix.")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive.")

    def x_cost(self, X: np.ndarray) -> float:
        """
        Penalize only tracking error e (first n entries of X).
        """
        X = np.asarray(X, dtype=float).reshape(-1)
        n = self.Q.shape[0]
        e = X[:n]
        return float(e.T @ self.Q @ e)

    def u_cost(self, u: np.ndarray) -> float:
        """
        Saturating control cost (closed form for diagonal R).
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        lam = self.lam

        # atanh requires |u/lam| < 1; clamp slightly inside bounds
        u_clamped = np.clip(u, -lam + 1e-9, lam - 1e-9)
        z = u_clamped / lam

        R = np.asarray(self.R, dtype=float)
        if not np.allclose(R, np.diag(np.diag(R))):
            raise ValueError("u_cost currently assumes R is diagonal.")
        rdiag = np.diag(R)

        term = z * np.arctanh(z) + 0.5 * safe_log(1.0 - z * z)
        U = 2.0 * np.sum((lam ** 2) * rdiag * term)
        return float(U)

    def running_cost(self, X: np.ndarray, u: np.ndarray) -> float:
        return self.x_cost(X) + self.u_cost(u)


# =========================================================
# sim/basis.py
# =========================================================

@dataclass(frozen=True)
class QuadraticPolynomialBasis:
    """
    Quadratic polynomial basis for an n-dimensional input X.

    phi(X) = [1, X_i, X_i^2, X_i*X_j for i<j]
    Provides Jacobian dphi/dX for gradient-based policies.
    """
    n: int

    @property
    def dim(self) -> int:
        return 1 + self.n + self.n + (self.n * (self.n - 1)) // 2

    def phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(self.n)
        feats: List[float] = [1.0]
        feats.extend(X.tolist())
        feats.extend((X * X).tolist())
        for i in range(self.n):
            for j in range(i + 1, self.n):
                feats.append(float(X[i] * X[j]))
        return np.asarray(feats, dtype=float)

    def dphi_dX(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(self.n)
        L = self.dim
        J = np.zeros((L, self.n), dtype=float)

        row = 0
        # constant
        row += 1

        # linear
        for i in range(self.n):
            J[row, i] = 1.0
            row += 1

        # squared
        for i in range(self.n):
            J[row, i] = 2.0 * X[i]
            row += 1

        # cross terms
        for i in range(self.n):
            for j in range(i + 1, self.n):
                J[row, i] = X[j]
                J[row, j] = X[i]
                row += 1

        return J


# =========================================================
# sim/irl_window.py
# =========================================================

@dataclass
class WindowSample:
    """One sample stored in the IRL rolling window."""
    t: float
    X: np.ndarray
    u: np.ndarray
    r: float


class IRLWindowBuffer:
    """
    Rolling buffer over last T seconds. Used to approximate the integral Bellman relation.
    """
    def __init__(self, T: float):
        if T <= 0.0:
            raise ValueError("Window length T must be positive.")
        self.T = float(T)
        self._buf: Deque[WindowSample] = deque()

    def append(self, sample: WindowSample) -> None:
        self._buf.append(sample)
        self._discard_old()

    def _discard_old(self) -> None:
        if not self._buf:
            return
        t_now = self._buf[-1].t
        t_min = t_now - self.T
        while self._buf and self._buf[0].t < t_min:
            self._buf.popleft()

    def is_ready(self) -> bool:
        if len(self._buf) < 2:
            return False
        return (self._buf[-1].t - self._buf[0].t) >= (0.95 * self.T)

    def get_endpoints(self) -> Tuple[WindowSample, WindowSample]:
        if len(self._buf) < 2:
            raise RuntimeError("Not enough samples in window.")
        return self._buf[0], self._buf[-1]

    def discounted_running_cost_integral(self, gamma: float) -> float:
        """
        Approximate:
          I = ∫ e^{-γ(τ-t0)} r(τ) dτ
        using trapezoidal rule on stored samples, discount anchored at window start t0.
        """
        if len(self._buf) < 2:
            return 0.0

        t0 = self._buf[0].t
        I = 0.0

        buf_list = list(self._buf)
        for i in range(len(buf_list) - 1):
            s_i = buf_list[i]
            s_j = buf_list[i + 1]

            dt = s_j.t - s_i.t
            if dt <= 0.0:
                continue

            wi = np.exp(-gamma * (s_i.t - t0))
            wj = np.exp(-gamma * (s_j.t - t0))
            I += 0.5 * dt * (wi * s_i.r + wj * s_j.r)

        return float(I)


# =========================================================
# sim/critic.py
# =========================================================

@dataclass
class IRLCritic:
    """
    Linear value approximation: V_hat(X) = W^T phi(X)
    Updated via integral Bellman error over a rolling window.
    """
    basis: object
    gamma: float
    alpha: float

    def __post_init__(self):
        if self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.W = np.zeros(self.basis.dim, dtype=float)

    def V_hat(self, X: np.ndarray) -> float:
        return float(self.W @ self.basis.phi(X))

    def gradV_hat(self, X: np.ndarray) -> np.ndarray:
        J = self.basis.dphi_dX(X)  # (L, nX)
        return J.T @ self.W        # (nX,)

    def update_from_window(self, window: IRLWindowBuffer, dt: float) -> dict:
        if not window.is_ready():
            return {"updated": False}

        start, end = window.get_endpoints()
        T_eff = end.t - start.t

        phi_start = self.basis.phi(start.X)
        phi_end = self.basis.phi(end.X)

        disc_T = float(np.exp(-self.gamma * T_eff))
        I = window.discounted_running_cost_integral(gamma=self.gamma)

        V_start = float(self.W @ phi_start)
        V_end = float(self.W @ phi_end)

        bellman_rhs = I + disc_T * V_end
        e_B = V_start - bellman_rhs

        delta_phi = phi_start - disc_T * phi_end
        denom = (1.0 + float(delta_phi @ delta_phi)) ** 2

        dW = -self.alpha * (delta_phi / denom) * e_B
        self.W = self.W + dt * dW

        return {
            "updated": True,
            "T_eff": T_eff,
            "I": float(I),
            "disc_T": disc_T,
            "V_start": V_start,
            "V_end": V_end,
            "e_B": float(e_B),
            "delta_phi_norm": float(np.linalg.norm(delta_phi)),
            "dW_norm": float(np.linalg.norm(dW)),
        }


# =========================================================
# sim/actor.py
# =========================================================

@dataclass
class IRLActor:
    """
    Saturated actor policy:
      u(X) = -lam * tanh( (1/(2*lam)) * R^{-1} * G^T * gradV_hat(X) )

    Here we approximate gradV_hat via actor weights W2:
      gradV_hat(X) = (dphi/dX)^T W2

    Initial implementation assumes scalar control (R shape 1x1), matching the SMD testbed.
    """
    basis: object
    R: np.ndarray
    lam: float
    m: float
    alpha: float

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=float)
        if self.R.shape != (1, 1):
            raise ValueError("Scalar control assumed: R must be shape (1,1).")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive.")
        if self.m <= 0.0:
            raise ValueError("mass m must be positive.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        self.W2 = np.zeros(self.basis.dim, dtype=float)
        self.R_inv = 1.0 / float(self.R[0, 0])

        # For augmented X=[e1,e2,xd1,xd2], input affects only e2dot via 1/m.
        self.G = np.array([0.0, 1.0 / self.m, 0.0, 0.0], dtype=float)  # (4,)

    def gradV_hat(self, X: np.ndarray) -> np.ndarray:
        J = self.basis.dphi_dX(X)     # (L, nX)
        return J.T @ self.W2          # (nX,)

    def policy(self, X: np.ndarray) -> np.ndarray:
        gradV = self.gradV_hat(X)
        s = (1.0 / (2.0 * self.lam)) * self.R_inv * float(self.G @ gradV)
        u = -self.lam * np.tanh(s)
        return np.array([u], dtype=float)

    def update_supervised(self, X: np.ndarray, u_target: np.ndarray, dt: float) -> dict:
        """
        Supervised actor update: match a target action u_target via gradient descent on:
          0.5 * (u_actor - u_target)^2
        """
        u_target = float(np.asarray(u_target).reshape(1)[0])
        u_actor = float(self.policy(X)[0])
        err = u_actor - u_target

        gradV = self.gradV_hat(X)
        s = (1.0 / (2.0 * self.lam)) * self.R_inv * float(self.G @ gradV)
        du_ds = -self.lam * (1.0 - np.tanh(s) ** 2)

        J = self.basis.dphi_dX(X)  # (L, nX)
        c = (1.0 / (2.0 * self.lam)) * self.R_inv
        ds_dW2 = c * (J @ self.G)  # (L,)

        dL_dW2 = err * du_ds * ds_dW2
        self.W2 = self.W2 - dt * self.alpha * dL_dW2

        return {
            "u_actor": u_actor,
            "u_target": u_target,
            "err": float(err),
            "dW2_norm": float(np.linalg.norm(dL_dW2)),
        }


# =========================================================
# sim/plot_run.py
# =========================================================

@dataclass(frozen=True)
class RunLogs:
    t: np.ndarray
    e: np.ndarray
    u: np.ndarray
    eB: np.ndarray
    eB_t: np.ndarray


def plot_logs(logs: RunLogs, title: str) -> None:
    t = logs.t

    plt.figure()
    plt.plot(t, logs.e[:, 0], label="e_pos")
    plt.plot(t, logs.e[:, 1], label="e_vel")
    plt.xlabel("t [s]")
    plt.ylabel("tracking error")
    plt.title(f"{title} - Tracking error")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t, logs.u[:, 0], label="u(t)")
    plt.xlabel("t [s]")
    plt.ylabel("control input")
    plt.title(f"{title} - Control input")
    plt.legend()
    plt.grid(True)

    if logs.eB.size > 0:
        plt.figure()
        plt.plot(logs.eB_t, logs.eB, label="|e_B|")
        plt.xlabel("t [s]")
        plt.ylabel("|e_B|")
        plt.title(f"{title} - Bellman error magnitude")
        plt.legend()
        plt.grid(True)

    plt.show()


def rms_over_tail(t: np.ndarray, signal: np.ndarray, tail_seconds: float) -> float:
    """
    RMS over the last `tail_seconds` of a signal.
    `signal` can be 1D or 2D (RMS over all elements in the tail).
    """
    t_end = float(t[-1])
    mask = t >= (t_end - tail_seconds)
    tail = signal[mask]
    return float(np.sqrt(np.mean(tail * tail)))


# =========================================================
# Shared control helpers (used by multiple runs)
# =========================================================

def pd_controller(x: np.ndarray, x_d: np.ndarray, kp: float, kd: float) -> np.ndarray:
    """PD baseline on tracking error e = x - x_d."""
    e = x - x_d
    return np.array([-kp * e[0] - kd * e[1]], dtype=float)


def greedy_action_from_critic(critic: IRLCritic, G: np.ndarray, R_inv: float, lam: float, X: np.ndarray) -> np.ndarray:
    """
    Greedy saturated action based on critic gradient (matches our paper implementation form).
    """
    gradV = critic.gradV_hat(X)
    s = (1.0 / (2.0 * lam)) * R_inv * float(G @ gradV)
    u = -lam * np.tanh(s)
    return np.array([u], dtype=float)


def linear_decay(t: float, t_start: float, t_end: float, a0: float) -> float:
    """Linear decay from a0 to 0 over [t_start, t_end]."""
    if t <= t_start:
        return a0
    if t >= t_end:
        return 0.0
    frac = (t_end - t) / (t_end - t_start)
    return a0 * frac


def ramp(t: float, t0: float, t1: float) -> float:
    """Linear ramp from 0 to 1 over [t0, t1]."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    return (t - t0) / (t1 - t0)


# =========================================================
# Runnable scripts (converted into functions)
# =========================================================

def run_pd_logged() -> None:
    plant = SpringMassDamper(m=1.0, c=0.5, k=2.0)
    dt = 0.001
    t_final = 10.0
    steps = int(t_final / dt)
    integrator = RK4(dt=dt)
    ref = SineReference(amplitude=1.0, omega=1.0)

    u_max = 2.0
    kp, kd = 10.0, 4.0

    t_log = np.zeros(steps)
    e_log = np.zeros((steps, 2))
    u_log = np.zeros((steps, 1))

    x = np.array([0.0, 0.0], dtype=float)
    t = 0.0

    for k in range(steps):
        x_d = ref.xd(t)
        u = pd_controller(x, x_d, kp, kd)
        u = clip(u, -u_max, u_max)

        t_log[k] = t
        e_log[k] = tracking_error(x, x_d)
        u_log[k] = u

        x = integrator.step(plant.dynamics, t, x, u)
        t += dt

    logs = RunLogs(t=t_log, e=e_log, u=u_log, eB=np.array([]), eB_t=np.array([]))

    print("PD baseline:")
    print(f"  RMS_pos(full): {np.sqrt(np.mean(e_log[:, 0] ** 2)):.6f}")
    print(f"  RMS_pos(last2s): {rms_over_tail(t_log, e_log[:, 0], 2.0):.6f}")
    print(f"  RMS_vel(last2s): {rms_over_tail(t_log, e_log[:, 1], 2.0):.6f}")
    print(f"  Max|u|: {float(np.max(np.abs(u_log))):.6f}")

    plot_logs(logs, title="PD baseline")


def run_irl_logged() -> None:
    plant = SpringMassDamper(m=1.0, c=0.5, k=2.0)
    dt = 0.001
    t_final = 10.0
    steps = int(t_final / dt)
    integrator = RK4(dt=dt)

    ref = SineReference(amplitude=1.0, omega=1.0)
    u_max = 2.0

    # Schedule params (as used in the project)
    kp, kd = 10.0, 4.0
    warmup_time = 4.0

    explore_a0 = 0.5
    explore_omega = 8.0
    explore_decay_start = 2.0
    explore_decay_end = 8.0

    q = np.diag([10.0, 1.0])
    r = np.diag([1.0])
    cost = ConstrainedTrackingCost(Q=q, R=r, lam=u_max)

    basis = QuadraticPolynomialBasis(n=4)
    gamma = 0.5
    window_T = 0.05
    critic_alpha = 10.0
    actor_alpha = 80.0

    window = IRLWindowBuffer(T=window_T)
    critic = IRLCritic(basis=basis, gamma=gamma, alpha=critic_alpha)
    actor = IRLActor(basis=basis, R=r, lam=u_max, m=plant.m, alpha=actor_alpha)

    R_inv = 1.0 / float(r[0, 0])

    t_log = np.zeros(steps)
    e_log = np.zeros((steps, 2))
    u_log = np.zeros((steps, 1))
    eB_log: List[float] = []
    eB_t_log: List[float] = []

    x = np.array([0.0, 0.0], dtype=float)
    t = 0.0

    for k in range(steps):
        x_d = ref.xd(t)
        X = augmented_state(x, x_d)

        u_actor = actor.policy(X)

        if t < warmup_time:
            u_base = pd_controller(x, x_d, kp, kd)
        else:
            u_base = u_actor

        a_t = linear_decay(t, explore_decay_start, explore_decay_end, explore_a0)
        u_explore = np.array([a_t * np.sin(explore_omega * t)], dtype=float)

        u = clip(u_base + u_explore, -u_max, u_max)

        t_log[k] = t
        e_log[k] = tracking_error(x, x_d)
        u_log[k] = u

        r_inst = cost.running_cost(X, u)
        window.append(WindowSample(t=t, X=X, u=u, r=r_inst))

        info = critic.update_from_window(window, dt=dt)
        if info.get("updated", False):
            eB_log.append(abs(info["e_B"]))
            eB_t_log.append(t)

        if window.is_ready():
            u_star = greedy_action_from_critic(critic, actor.G, R_inv, u_max, X)
            actor.update_supervised(X, u_target=u_star, dt=dt)

        x = integrator.step(plant.dynamics, t, x, u)
        t += dt

    logs = RunLogs(
        t=t_log,
        e=e_log,
        u=u_log,
        eB=np.asarray(eB_log, dtype=float),
        eB_t=np.asarray(eB_t_log, dtype=float),
    )

    print("IRL scheduled run:")
    print(f"  RMS_pos(full): {np.sqrt(np.mean(e_log[:, 0] ** 2)):.6f}")
    print(f"  RMS_pos(last2s): {rms_over_tail(t_log, e_log[:, 0], 2.0):.6f}")
    print(f"  RMS_vel(last2s): {rms_over_tail(t_log, e_log[:, 1], 2.0):.6f}")
    print(f"  Max|u|: {float(np.max(np.abs(u_log))):.6f}")
    print(f"  Mean|eB|: {float(np.mean(np.abs(logs.eB))):.6f}" if logs.eB.size else "  Mean|eB|: n/a")

    plot_logs(logs, title="IRL scheduled (warm-start + decay exploration)")


def run_irl_logged_blend() -> None:
    plant = SpringMassDamper(m=1.0, c=0.5, k=2.0)

    dt = 0.001
    t_final = 12.0
    steps = int(t_final / dt)
    integrator = RK4(dt=dt)

    ref = SineReference(amplitude=1.0, omega=1.0)
    u_max = 2.0

    # PD stabilizing partner
    kp, kd = 10.0, 4.0

    # Exploration schedule
    explore_a0 = 0.5
    explore_omega = 8.0
    explore_decay_start = 2.0
    explore_decay_end = 8.0

    # Blending schedule (PD -> actor)
    blend_start = 2.0
    blend_end = 12.0
    beta_max = 0.6  # best from sweep

    # Cost + learning params
    q = np.diag([10.0, 1.0])
    r = np.diag([1.0])
    cost = ConstrainedTrackingCost(Q=q, R=r, lam=u_max)

    basis = QuadraticPolynomialBasis(n=4)
    gamma = 0.5
    window_T = 0.05
    critic_alpha = 10.0
    actor_alpha = 80.0

    window = IRLWindowBuffer(T=window_T)
    critic = IRLCritic(basis=basis, gamma=gamma, alpha=critic_alpha)
    actor = IRLActor(basis=basis, R=r, lam=u_max, m=plant.m, alpha=actor_alpha)

    R_inv = 1.0 / float(r[0, 0])

    # Logs
    t_log = np.zeros(steps)
    e_log = np.zeros((steps, 2))
    u_log = np.zeros((steps, 1))
    eB_log: List[float] = []
    eB_t_log: List[float] = []

    x = np.array([0.0, 0.0], dtype=float)
    t = 0.0

    for k in range(steps):
        x_d = ref.xd(t)
        X = augmented_state(x, x_d)

        u_pd = pd_controller(x, x_d, kp, kd)
        u_actor = actor.policy(X)

        beta = min(ramp(t, blend_start, blend_end), beta_max)
        u_base = (1.0 - beta) * u_pd + beta * u_actor

        a_t = linear_decay(t, explore_decay_start, explore_decay_end, explore_a0)
        u_explore = np.array([a_t * np.sin(explore_omega * t)], dtype=float)

        u = clip(u_base + u_explore, -u_max, u_max)

        t_log[k] = t
        e_log[k] = tracking_error(x, x_d)
        u_log[k] = u

        r_inst = cost.running_cost(X, u)
        window.append(WindowSample(t=t, X=X, u=u, r=r_inst))

        info = critic.update_from_window(window, dt=dt)
        if info.get("updated", False):
            eB_log.append(abs(info["e_B"]))
            eB_t_log.append(t)

        if window.is_ready():
            u_star = greedy_action_from_critic(critic, actor.G, R_inv, u_max, X)
            actor.update_supervised(X, u_target=u_star, dt=dt)

        x = integrator.step(plant.dynamics, t, x, u)
        t += dt

    logs = RunLogs(
        t=t_log,
        e=e_log,
        u=u_log,
        eB=np.asarray(eB_log, dtype=float),
        eB_t=np.asarray(eB_t_log, dtype=float),
    )

    print("IRL BLENDED FINAL run (beta_max=0.60):")
    print(f"  RMS_pos(full): {np.sqrt(np.mean(e_log[:, 0] ** 2)):.6f}")
    print(f"  RMS_pos(last2s): {rms_over_tail(t_log, e_log[:, 0], 2.0):.6f}")
    print(f"  RMS_vel(last2s): {rms_over_tail(t_log, e_log[:, 1], 2.0):.6f}")
    print(f"  Max|u|: {float(np.max(np.abs(u_log))):.6f}")
    print(f"  Mean|eB|: {float(np.mean(np.abs(logs.eB))):.6f}" if logs.eB.size else "  Mean|eB|: n/a")

    plot_logs(logs, title="IRL blended FINAL (beta_max=0.60)")


def sweep_beta_max() -> None:
    """
    Sweep actor authority cap (beta_max) for the blended controller,
    printing a compact metrics table.
    """
    def run_once(beta_max: float) -> Dict[str, float]:
        plant = SpringMassDamper(m=1.0, c=0.5, k=2.0)
        dt = 0.001
        t_final = 12.0
        steps = int(t_final / dt)
        integrator = RK4(dt=dt)

        ref = SineReference(amplitude=1.0, omega=1.0)
        u_max = 2.0

        kp, kd = 10.0, 4.0

        explore_a0 = 0.5
        explore_omega = 8.0
        explore_decay_start = 2.0
        explore_decay_end = 8.0

        blend_start = 2.0
        blend_end = 12.0

        q = np.diag([10.0, 1.0])
        r = np.diag([1.0])
        cost = ConstrainedTrackingCost(Q=q, R=r, lam=u_max)

        basis = QuadraticPolynomialBasis(n=4)
        gamma = 0.5
        window_T = 0.05
        critic_alpha = 10.0
        actor_alpha = 80.0

        window = IRLWindowBuffer(T=window_T)
        critic = IRLCritic(basis=basis, gamma=gamma, alpha=critic_alpha)
        actor = IRLActor(basis=basis, R=r, lam=u_max, m=plant.m, alpha=actor_alpha)
        R_inv = 1.0 / float(r[0, 0])

        t_log = np.zeros(steps)
        e_log = np.zeros((steps, 2))
        u_log = np.zeros((steps, 1))
        eB_abs: List[float] = []

        x = np.array([0.0, 0.0], dtype=float)
        t = 0.0

        for k in range(steps):
            x_d = ref.xd(t)
            X = augmented_state(x, x_d)

            u_pd = pd_controller(x, x_d, kp, kd)
            u_actor = actor.policy(X)

            beta = min(ramp(t, blend_start, blend_end), beta_max)
            u_base = (1.0 - beta) * u_pd + beta * u_actor

            a_t = linear_decay(t, explore_decay_start, explore_decay_end, explore_a0)
            u_explore = np.array([a_t * np.sin(explore_omega * t)], dtype=float)
            u = clip(u_base + u_explore, -u_max, u_max)

            t_log[k] = t
            e_log[k] = tracking_error(x, x_d)
            u_log[k] = u

            r_inst = cost.running_cost(X, u)
            window.append(WindowSample(t=t, X=X, u=u, r=r_inst))

            info = critic.update_from_window(window, dt=dt)
            if info.get("updated", False):
                eB_abs.append(abs(info["e_B"]))

            if window.is_ready():
                u_star = greedy_action_from_critic(critic, actor.G, R_inv, u_max, X)
                actor.update_supervised(X, u_target=u_star, dt=dt)

            x = integrator.step(plant.dynamics, t, x, u)
            t += dt

        return {
            "beta_max": float(beta_max),
            "rms_pos_last2s": float(rms_over_tail(t_log, e_log[:, 0], 2.0)),
            "rms_vel_last2s": float(rms_over_tail(t_log, e_log[:, 1], 2.0)),
            "max_u": float(np.max(np.abs(u_log))),
            "mean_abs_eB": float(np.mean(eB_abs)) if len(eB_abs) else float("nan"),
        }

    betas = [0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0]
    print("beta_max\tRMS_pos(last2s)\tRMS_vel(last2s)\tmax|u|\tmean|eB|")
    for b in betas:
        res = run_once(beta_max=b)
        print(
            f"{res['beta_max']:.2f}\t\t"
            f"{res['rms_pos_last2s']:.6f}\t\t"
            f"{res['rms_vel_last2s']:.6f}\t\t"
            f"{res['max_u']:.6f}\t"
            f"{res['mean_abs_eB']:.6f}"
        )


# =========================================================
# CLI entry point
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Single-file runner for the IRL spring–mass–damper project.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run_pd", help="Run PD baseline with logging + plots.")
    sub.add_parser("run_irl", help="Run IRL scheduled controller with logging + plots.")
    sub.add_parser("run_blend", help="Run blended PD+IRL controller (beta_max=0.6) with plots.")
    sub.add_parser("sweep_beta", help="Sweep beta_max values and print metrics table (no plots).")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "run_pd":
        run_pd_logged()
    elif args.cmd == "run_irl":
        run_irl_logged()
    elif args.cmd == "run_blend":
        run_irl_logged_blend()
    elif args.cmd == "sweep_beta":
        sweep_beta_max()
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
