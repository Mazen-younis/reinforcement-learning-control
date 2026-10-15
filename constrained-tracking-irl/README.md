# Integral Reinforcement Learning for Constrained Tracking Control

A simulation-based investigation of Integral Reinforcement Learning (IRL) for constrained trajectory tracking, comparing classical Proportional-Derivative (PD) control with an actor–critic reinforcement learning approach.

**Authors:** Mazen Younis · Mert Satici  
**Institution:** Berliner Hochschule für Technik (BHT)  
**Course:** Reinforcement Learning  
**Year:** 2026

## Overview

This project investigates how reinforcement learning can be applied to a continuous-time tracking control problem with actuator constraints.

A second-order dynamical system must follow a time-varying reference trajectory while respecting limits on the applied control input.

Three control approaches are investigated:

1. Classical PD controller
2. Integral Reinforcement Learning using an actor–critic architecture
3. Blended PD + IRL control

## Objectives

- Minimise position and velocity tracking errors.
- Respect actuator saturation constraints.
- Investigate learning-based control with actor–critic methods.
- Compare reinforcement learning against a classical PD baseline.
- Evaluate the benefits and limitations of blended control.

## Project Documentation

The Python implementation, experimental results and complete project report will be added to this directory.
