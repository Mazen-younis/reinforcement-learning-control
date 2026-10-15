# Reinforcement Learning for Control Systems

**Simulation-based reinforcement learning for DC motor speed control and constrained trajectory tracking.**

**Authors:** Mazen Younis · Mert Satici  
**Institution:** Berliner Hochschule für Technik (BHT) · Computer Engineering  
**Course:** Reinforcement Learning  
**Year:** 2026

## Overview

This repository presents **two completed university engineering projects** that investigate reinforcement learning (RL) for control tasks. Both use simulated dynamical systems, but they address different problems and learning methods:

| Project | Simulated system | Methods | Main question |
|---|---|---|---|
| **[DC Motor Speed Control](dc-motor-sarsa/README.md)** | Electrical and mechanical DC motor dynamics | Tabular SARSA(0), ε-greedy exploration | Can a learned voltage policy track target speed under different loads? |
| **[Constrained Trajectory Tracking](constrained-tracking-irl/README.md)** | Spring–mass–damper system | PD baseline, integral actor–critic RL, blended PD–IRL | Does a learning-based controller improve on classical PD control when actuator input is limited? |

Both projects include **Python implementations, academic reports, selected experimental figures, and technical documentation**. Their results were obtained in simulation, not on physical hardware.

## Project 1 — DC Motor Speed Control with SARSA

A physics-based DC motor simulation provides the environment for a tabular **SARSA(0)** controller. The agent learns to select discrete, bounded voltage commands to track an angular-speed reference while limiting overshoot and responding to varying load torque.

**Technical focus:** motor dynamics · state discretisation · on-policy temporal-difference learning · reward shaping · ε-greedy exploration · evaluation across load conditions

### Reported performance

| Load | Rise time to 90% | Overshoot | Steady-state error |
|---|---:|---:|---:|
| No load | 0.78 ms | 0.97% | 0.059 rad/s |
| Nominal load | 0.80 ms | 0.94% | −0.087 rad/s |
| High load | 0.84 ms | 0.91% | 0.404 rad/s |

The submitted report shows that the evaluated policy met the assignment's **≤ 1 ms rise-time** and **≤ 3% overshoot** requirements in all three scenarios.

![DC motor greedy-policy speed tracking for three load conditions](dc-motor-sarsa/results/speed_tracking_across_loads.png)

**Explore:** [Technical README](dc-motor-sarsa/README.md) · [Python source](dc-motor-sarsa/main.py) · [Results gallery](dc-motor-sarsa/results/README.md) · [Full report (PDF)](dc-motor-sarsa/DC_Motor_Speed_Control_Report.pdf)

> **Reproducibility:** These values and figures are from the submitted course report, not independently regenerated for this GitHub release. The published source includes additional training modes and differences from the report's experiment configuration; consult the [project README](dc-motor-sarsa/README.md) before comparing runs.

## Project 2 — Integral Reinforcement Learning for Constrained Tracking

A simulated **spring–mass–damper system** follows a time-varying reference trajectory subject to bounded actuator force. The study compares a classical **proportional–derivative (PD)** controller, an **integral actor–critic reinforcement learning** controller, and a **blended PD–IRL** strategy.

**Technical focus:** optimal control · actor–critic learning · integral Bellman updates · quadratic feature approximation · continuous control · actuator saturation · controller benchmarking

### Reported controller comparison

| Metric | PD baseline | Blended PD + IRL |
|---|---:|---:|
| RMS position error (last 2 s) | **0.0537** | 0.1781 |
| RMS velocity error (last 2 s) | **0.0772** | 0.1053 |
| Maximum absolute control input | 2.0 | 2.0 |
| Mean Bellman-error magnitude | — | 0.0424 |

**Key finding:** In this test, the conventional **PD baseline achieved lower tracking errors** than the blended learning-based controller. The project illustrates why RL controllers need to be assessed against meaningful classical baselines rather than judged only on learning progress.

![Position and velocity tracking errors for the blended PD–IRL controller](constrained-tracking-irl/results/blended_irl_tracking_error.png)

**Explore:** [Technical README](constrained-tracking-irl/README.md) · [Python source](constrained-tracking-irl/main.py) · [Results gallery](constrained-tracking-irl/results/README.md) · [Full report (PDF)](constrained-tracking-irl/Integral_Reinforcement_Learning_Report.pdf)

> **Comparison note:** The source uses different simulation durations for the PD and blended runs (10 s and 12 s), meaning the reported final-two-second error windows are not identical. The [project README](constrained-tracking-irl/README.md) details this and other reproducibility considerations.

## Repository structure

```text
reinforcement-learning-control/
├── README.md
├── dc-motor-sarsa/
│   ├── README.md
│   ├── main.py
│   ├── DC_Motor_Speed_Control_Report.pdf
│   └── results/
│       ├── README.md
│       ├── speed_tracking_across_loads.png
│       ├── training_progress_nominal_1000_episodes.png
│       └── control_signal_nominal_load.png
└── constrained-tracking-irl/
    ├── README.md
    ├── main.py
    ├── Integral_Reinforcement_Learning_Report.pdf
    └── results/
        ├── README.md
        ├── pd_tracking_error.png
        ├── blended_irl_tracking_error.png
        └── irl_bellman_error.png
```

## Technologies and running the projects

- **Language:** Python 3
- **Scientific libraries:** NumPy and Matplotlib
- **Learning methods:** SARSA(0), temporal-difference learning, integral reinforcement learning, actor–critic methods
- **Control and simulation:** DC motor dynamics, spring–mass–damper dynamics, RK4 integration, PD control, constrained control actions

Each project is **independent** and provides its own command-line examples, configuration notes, and limitations in its technical README:

- [Run the DC motor SARSA project](dc-motor-sarsa/README.md#run-the-implementation)
- [Run the constrained-tracking IRL project](constrained-tracking-irl/README.md#run-the-implementation)

Other course labs and exploratory scripts are intentionally excluded so this repository focuses on the two complete engineering studies.

## Scope and limitations

These are **academic simulation studies**. Neither project demonstrates physical-hardware deployment, real-time performance guarantees, or production safety validation. Reported metrics are tied to the original experiment settings and should not be interpreted as fresh benchmark results from the current published scripts.

## Attribution

Both university projects were completed collaboratively by **Mazen Younis and Mert Satici** at **Berliner Hochschule für Technik (BHT)** in 2026. The code, reports, and selected results are presented for technical portfolio and educational purposes.
