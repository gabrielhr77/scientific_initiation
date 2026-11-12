# Neural Prediction of Gravitational Dynamics

**Scientific initiation research · University of São Paulo (USP) · FAPESP-supported**

An investigation of how neural networks predict the evolution of gravitational **two- and three-body systems**, and how their predictions compare with numerical integration. The work combines synthetic trajectory generation, PyTorch models, autoregressive evaluation and physical-consistency analysis.

> **Repository status: research code, being prepared for reproducibility.** The ten research Python scripts are included. `saturn2.py` was reconstructed from a Markdown-formatted paste (without intentional algorithmic changes). Simulation datasets and model checkpoints are **not included** in this package. Do not treat the current bundle as a complete reproduction of the experiments.

## Research focus

The models predict the next state of gravitating bodies (positions and velocities) from their current state or a window of prior states, then can be evaluated by repeated autoregressive predictions. The project studies one-step accuracy, error accumulation, prediction horizon, energy drift, momentum conservation and sensitivity to perturbations.

## Pipeline

```text
Initial conditions -> 4th-order Yoshida integrator -> synthetic trajectories (.npz)
                 -> PyTorch neural models -> checkpoints
                 -> 1-step and autoregressive diagnostics -> plots and metrics
```

## Implemented components in this bundle

| Component | Files | Purpose |
|---|---|---|
| Numerical physics | `fisica2corpos.py`, `fisica3corpos.py` | Fourth-order Yoshida integration, accelerations and physical quantities (NumPy). |
| Dataset generation | `config.py`, `preparar_dataset.py`, `geradorDataset.py` | Seeded simulation generation and compressed NPZ output. |
| Two-body neural model | `saturn3b.py` | Body-token Transformer training script with physics-informed feature engineering. |
| Three-body neural model | `saturn3.py` | Three-body body-token Transformer training script. |
| Evaluation | `diagnostico.py`, `diagnostico2C.py` | Rollouts, horizons, physical metrics, baselines and diagnostic plots. |

**Important naming note:** `saturn3b.py` is the **two-body** model; `saturn3.py` is the **three-body** model. The shared legacy module `saturn2.py` is now included; compatibility across the different model versions still needs testing. Their names are preserved to avoid hiding the historical architecture of this research.

The newer body-token models use the **latest state** to predict the next state, while the diagnostic tools retain a configurable input window for compatibility with other checkpoints. This distinction matters when reading the evolution of the project.

## Research methods

- Generate initial conditions with randomized masses, positions and velocities; normalize the system to a common energy scale and reject near encounters or unbound/escaping scenarios according to the configuration.
- Integrate the gravitational motion using a fourth-order Yoshida symplectic integrator and export trajectory data in NPZ format.
- Learn state increments with PyTorch Transformer encoders, relational features between bodies, AdamW, learning-rate scheduling and exponential moving averages of weights.
- Evaluate one-step errors and long-horizon autoregressive predictions against numerical integration and reference trajectories.

## Preliminary results from the research report draft

The numbers below are taken from the **draft research report**, not rerun or verified from checkpoints in this bundle. One-step errors are measured using the report's normalized `escalaDelta` scale. These results may correspond to a specific experiment or checkpoint; they should not be assumed to describe every model version.

| Metric | Two bodies | Three bodies |
|---|---:|---:|
| One-step position RMS error | 0.0004 | 0.0009 |
| One-step velocity RMS error | 0.0008 | 0.0021 |
| Median autoregressive prediction horizon | 520 steps | 560 steps |

The horizon is specific to the metric and threshold used in the draft report and an evaluation capped at 1,000 steps. Long-horizon predictions accumulated systematic error and energy drift. The neural models **did not outperform** the fourth-order Yoshida baseline in long-horizon precision in those experiments.

## Setup and status of execution

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m compileall -q .
```

**Training/diagnostics are not yet independently validated** with this package: generated NPZ datasets and trained checkpoints are missing, and some cross-version interfaces require fixes. Once the shared dependency has been added and validated, the scripts provide these entry points:

```bash
# Example command defined by saturn3.py (requires datasets and saturn2.py)
python saturn3.py --passos 100000

# Example command defined by saturn3b.py (requires datasets and saturn2.py)
python saturn3b.py --passos 100000
```

The `geradorDataset.py` script currently uses a hard-coded `selecao` parameter and aims to produce thousands of trajectories. Review `config.py` and resource usage before invoking it; it is **not** a quick-start demo. The diagnostics require checkpoints with compatible architectures/configuration and their respective datasets.

See [Code map and known gaps](docs/CODE_MAP.md) before restructuring imports.

## Data format

The dataset export function records `massas`, `estado`, `tempo`, `energia`, `momAng`, `momLin`, `dt` and `motivoTermino` in compressed `.npz` files. The original `estado` rows include the simulation time as an extra final column. Models load the first 12 (two bodies) or 18 (three bodies) physical state components as needed.

## Limitations and roadmap

- Test and repair cross-version compatibility, including the `escalaDelta` variable referenced by `saturn2.RedeTransformer.__init__` but only assigned inside the training block.
- Check the correspondence between checkpoints, model version and result figures.
- Provide reproducible small dataset generation and a CPU-friendly demonstration.
- Pin dependencies from a tested environment; run unit and integration tests.
- Add representative plots with the exact experiment and evaluation definitions.
- Document the differences between earlier temporal-window models and the later body-token models.

## Academic attribution

**Researcher:** Gabriel Hupfer Righi  
**Institution:** University of São Paulo (USP), Ribeirão Preto  
**Advisor:** Prof. Dr. Luiz Otávio Murta Junior  
**Support:** São Paulo Research Foundation (FAPESP)

The repository's public release, reuse rights, and any potential license require confirmation with the research supervisor and applicable institutional/funding requirements. No software license has been added by default.

## Links

[GitHub profile](https://github.com/gabrielhr77) · [LinkedIn](https://www.linkedin.com/in/gabriel-hupfer-righi-915064367/)
