# Neural Prediction of Gravitational Dynamics

**Undergraduate Research | University of São Paulo (USP), Ribeirão Preto | FAPESP-supported**

This repository documents research on neural-network prediction of gravitational two- and three-body dynamics. The project combines numerical simulation, synthetic trajectory datasets, deep learning with PyTorch, and evaluation of physical consistency and long-term prediction errors.

> **Status:** Research repository under documentation and reproducibility review. Not all scripts currently form a single validated end-to-end workflow.

## Research objectives

- Generate reference trajectories for gravitational systems using numerical integration.
- Train neural networks to predict future positions and velocities.
- Compare one-step predictions and autoregressive rollouts with numerical baselines.
- Analyze prediction horizons and changes in energy, linear momentum, and angular momentum.

## Repository structure

```text
scientific_initiation/
├── README.md
├── .gitignore
├── dados/                         # Early experiments and sample data
├── ESTUDOS DE REDES-NEURAIS/     # Neural-network studies
├── REDES_NO_SERVIDOR/            # Server-side experiments
├── projeto/                       # Main research implementation
│   ├── config.py
│   ├── fisica2corpos.py
│   ├── fisica3corpos.py
│   ├── geradorDataset.py
│   ├── preparar_dataset.py
│   ├── conversoesEstado.py
│   ├── modelo.py
│   ├── loss.py
│   ├── treino.py
│   ├── finalizacao.py
│   ├── saturn3.py
│   ├── saturn3b.py
│   ├── diagnostico2C.py
│   ├── versoes_treino_redes/
│   │   ├── saturn1.py
│   │   └── saturn2.py
│   └── simulacoesArtificiais/
│       ├── simulacoes2C/
│       └── simulacoes3C/
└── [earlier simulation and regression scripts]
```

This structure reflects the repository's current layout, including exploratory work. Files and directories may be reorganized after import dependencies and execution paths are tested.

## Methods

**Numerical reference:** synthetic gravitational trajectories, including implementations of a fourth-order Yoshida symplectic integrator.

**Machine learning:** PyTorch-based neural models, including Transformer experiments for next-state prediction. The code includes trajectory normalization, windowed datasets, autoregressive rollout evaluation, and losses based on state error and physical quantities.

**Evaluation:** one-step error, comparison with persistence and numerical baselines, rollout error, predictability horizon, and physical diagnostics.

## Preliminary research findings

The research draft reports the following preliminary normalized one-step errors:

| Metric | Two-body | Three-body |
|---|---:|---:|
| Position RMS error | 0.0004 | 0.0009 |
| Velocity RMS error | 0.0008 | 0.0021 |
| Median prediction horizon | 520 steps | 560 steps |

These values refer to the research draft's evaluation protocol and should not be interpreted as universal performance guarantees. The neural predictors did **not** outperform the Yoshida numerical integrator in long-horizon accuracy. Computational speed comparisons have not yet been established.

## Reproducibility status

The repository includes research scripts and some sample `.npz` trajectories, but the full experiment environment, trained checkpoints, and execution instructions have not yet been independently validated. Some historical model versions require compatibility fixes. Detailed installation and run commands will be added after a clean-environment test.

## Technology

Python · PyTorch · NumPy · SciPy · Matplotlib

## Research context

**Researcher:** Gabriel Hupfer Righi  
**Institution:** University of São Paulo (USP), Ribeirão Preto, Brazil  
**Support:** São Paulo Research Foundation (FAPESP)

[GitHub profile](https://github.com/gabrielhr77) · [LinkedIn](https://www.linkedin.com/in/gabriel-hupfer-righi-915064367/)

## Notes

- The presence of a script does not imply that it is the final model used in the reported experiments.
- Dataset and checkpoint publication should be reviewed for size, reproducibility, and applicable permissions.
- No license has been selected yet; reuse rights should not be assumed.
