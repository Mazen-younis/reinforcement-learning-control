# Reinforcement Learning for Control Systems

**Simulation-based reinforcement learning for DC motor speed control and constrained trajectory tracking.**

**Authors:** Mazen Younis · Mert Satici  
**Institution:** Berliner Hochschule für Technik (BHT) · Computer Engineering  
**Course:** Reinforcement Learning  
**Year:** 2026

## Overview

This repository presents two complementary university projects investigating how reinforcement learning can be applied to **engineering control problems**. Both studies use simulated dynamical systems and evaluate learning-based controllers against explicit performance or actuator constraints.

The projects cover two different control settings:

1. **DC motor speed control:** learning bounded voltage commands with a tabular SARSA agent and evaluating speed tracking under changing loads.
2. **Constrained trajectory tracking:** investigating an integral reinforcement learning (IRL) actor–critic controller alongside a classical proportional–derivative (PD) baseline and a blended PD–IRL approach.

The aim is to examine **how learning algorithms behave in control applications**, what their limitations are, and how performance should be assessed beyond training rewards alone.

## Projects

### 1. DC Motor Speed Control with SARSA

**Status:** Published · [Explore the project](dc-motor-sarsa/)

A physics-based DC motor simulation is used to train a **tabular SARSA(0)** controller to select discrete armature voltages. The control objective is to reach a target angular speed quickly while limiting overshoot, respecting voltage constraints, and maintaining performance across different load conditions.

**Technical focus:** DC motor dynamics · SARSA(0) · ε-greedy exploration · reward shaping · constrained actions · robustness evaluation

**Selected results from the submitted course report:**

| Load condition | Rise time to 90% | Overshoot | Steady-state error |
|---|---:|---:|---:|
| No load | 0.78 ms | 0.97% | 0.059 rad/s |
| Nominal load | 0.80 ms | 0.94% | −0.087 rad/s |
| High load | 0.84 ms | 0.91% | 0.404 rad/s |

All three **reported** scenarios satisfied the assignment's rise-time (≤ 1 ms) and overshoot (≤ 3%) thresholds.

**Explore:** [Technical README](dc-motor-sarsa/README.md) · [Python implementation](dc-motor-sarsa/main.py) · [Selected result figures](dc-motor-sarsa/results/) · [Project report (PDF)](dc-motor-sarsa/DC_Motor_Speed_Control_Report.pdf)

> **Reproducibility note:** These figures come from the submitted report. The published Python source supports additional training configurations and does not, by default, guarantee reproduction of the exact reported values. The project README explains the differences.

### 2. Integral Reinforcement Learning for Constrained Tracking Control

**Status:** Being prepared for publication

This study considers a simulated **second-order dynamical system** tracking a time-varying reference under bounded control input. It investigates three approaches: a **classical PD controller**, an **IRL actor–critic controller**, and a **blended PD + IRL strategy**.

**Technical focus:** constrained optimal control · actor–critic learning · continuous-time dynamics · trajectory tracking · actuator saturation · baseline comparison

An important finding is that the classical PD baseline achieved lower steady-state tracking errors than the blended IRL controller in the reported test. The work therefore highlights practical trade-offs, rather than claiming that reinforcement learning always outperforms classical control.

The source code, full report, evaluation results, and project-specific README will be linked here as they are reviewed and published.

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
│       └── [selected training and control figures]
└── constrained-tracking-irl/    # In preparation
```

Additional course exercises are **not** included by default: this repository is curated around the two larger engineering studies rather than serving as a dump of every lab assignment.

## Technologies and methods

**Language:** Python  
**Scientific computing:** NumPy, Matplotlib  
**Reinforcement learning:** SARSA, temporal-difference learning, ε-greedy exploration, integral reinforcement learning, actor–critic methods  
**Control and evaluation:** DC motor simulation, PD control, reward design, state discretisation, trajectory tracking, input saturation, load disturbances

Each project directory contains its own technical documentation and implementation details. Dependencies and reproduction instructions should be checked **per project** rather than assumed to be identical.

## Scope and limitations

These are **simulation-based academic studies**, not controllers validated on physical industrial hardware. Reported results are tied to the documented model, scenarios, and training/evaluation settings. They do not establish real-world safety, stability, or production readiness.

## Attribution

This coursework was completed collaboratively by **Mazen Younis and Mert Satici** at **Berliner Hochschule für Technik (BHT)**. The repository documents the work for educational and technical portfolio purposes.
