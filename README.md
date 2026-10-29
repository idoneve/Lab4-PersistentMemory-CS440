# Vacuum World — Context Engineering Strategies 1 & 4

## Overview

Implementation of the AIMA vacuum world (4x4 grid) controlled by two
different context-engineering agent architectures:

- **Strategy 1**: Persistent memory architecture
- **Strategy 4**: Isolation via sub-agent

Q-learning will be added in a later lab; this repository contains the
environment and agent scaffolding only.

## Environment

4x4 grid, initial state:

start 0 0 0
  0   X 0 0
  0   0 0 0
  0  -1 0 10

- `start` = initial robot position (0,0)
- `X` = obstacle (impassable)
- `-1` = penalty cell
- `10` = reward cell
- `0` = plain cell

Actions: `UP`, `DOWN`, `LEFT`, `RIGHT`, `SUCK`, `NO-OP`.

## Project Structure

.
├── README.md
├── spec.md # Specification file handed to the LLM
├── requirements.txt
├── bootstrap.ipynb
├── vacuum_world_lab4.ipynb # Strategy 1: persistent memory agent
├── /Test # Vacuum world tests
└── /aima # AIMA repository

### Strategy 1 — Persistent Memory (`agent_strategy1.py`)

The agent keeps an internal dictionary of everything it has learned
about the grid (visited cells, known obstacles, known rewards). Decisions
are made from `current percept + stored memory`, so the agent does not
re-explore known cells and does not loop forever.

## How to Run

Use vacuum_world_lab4.ipynb for the code to run tests in /Test