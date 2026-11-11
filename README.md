# GO-IRSA
Goal-oriented Irregular Repetition Slotted ALOHA

## Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

## Usage
Run the adaptive target load policy and compare it with the pull-based policy.

```bash
options:
  -h, --help            show this help message and exit
  --nodes NODES         Number of nodes in the system
  --slot_per_frame SLOT_PER_FRAME
                        Number of slots per frame
  --dim DIM             Dimension of the process (1 or 2)
  --n_bins N_BINS       Number of bins for the belief distribution
  --sigma SIGMA         Standard deviation for the initial distribution
  --episodes EPISODES   Number of episodes to run
  --steps STEPS         Number of slots to simulate
  --device DEVICE       Device to run the simulation on (cuda or cpu)
  --target_loads TARGET_LOADS [TARGET_LOADS ...]
                        List of target loads to test the TargetIRSA policy with
  --irsa_model {optimal,imperfect}
                        IRSA model to use for the TargetIRSA policy (optimal or imperfect)
  --degree_dist DEGREE_DIST
                        Degree distribution to use for the TargetIRSA policy (bcsa or liva)
  --num_subcarriers NUM_SUBCARRIERS
                        Number of subcarriers to use
```

Run the RL-based policy for IRSA

NOTE: The RL-based policy uses `stable-baselines3` for training the agent.

```bash
options:
  -h, --help            show this help message and exit
  --nodes NODES         Number of nodes in the system
  --slot_per_frame SLOT_PER_FRAME
                        Number of slots per frame
  --dim DIM             Dimension of the process (1 or 2)
  --n_bins N_BINS       Number of bins for the belief distribution
  --sigma SIGMA         Standard deviation for the initial distribution
  --episodes EPISODES   Number of episodes to run
  --steps STEPS         Total number of slots to simulate
  --device DEVICE       Device to run the simulation on (cuda or cpu)
  --irsa_model {optimal,imperfect}
                        IRSA model to use for the TargetIRSA policy (optimal or imperfect)
  --degree_dist DEGREE_DIST
                        Degree distribution to use for the TargetIRSA policy (bcsa or liva)
  --num_subcarriers NUM_SUBCARRIERS
                        Number of subcarriers to use
  --training_steps TRAINING_STEPS
                        Number of training steps for the RL agent
  --num_envs NUM_ENVS   Number of parallel environments for training
```

## Performance note
While the scripts runs without strict GPU requirement, using a GPU can significantly speed up the simulation process, especially when the number of nodes and slots per frame is large.
## Count-conditioned beliefs and online load calibration (`oracle_study/`)

Extension of GO-IRSA with (i) a belief update conditioned on the number of undecoded
transmitters, estimated from the residual power after SIC, and (ii) an online
calibration of the load target. Report: `oracle_study/report/report.tex`.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv numpy scipy numba matplotlib pandas pyarrow zipfile-deflate64
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu121

# Wiener model (paper scenario): conservative vs estimated, with / without calibration
.venv/bin/python -m oracle_study.run_gpu --episodes 50 --eta 1.0 0.95
.venv/bin/python -m oracle_study.run_gpu --episodes 50 --eta 1.0 0.95 --calib_gain 0.3 --tag _calib --no_pull

# Real data: Low Carbon London smart meters (one sample = one frame, AR(1) per household)
.venv/bin/python -m oracle_study.realdata.prepare     # needs data/lcl/LCL-FullData.zip (London Datastore)
.venv/bin/python -m oracle_study.realdata.fit
.venv/bin/python -m oracle_study.realdata.run_real --episodes 50 --G 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
.venv/bin/python -m oracle_study.realdata.run_real --episodes 50 --G 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
    --calib_gain 0.3 --no_pull --out oracle_study/results/real_hold_episodes_calib.csv

# Report figures / tables
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m oracle_study.report.trace_calib
.venv/bin/python -m oracle_study.report.make_figures
```

| File | Content |
|---|---|
| `oracle_study/sim.py` | config, IRSA frame with SIC, residual-power estimator of K_u, pull baseline |
| `oracle_study/sim_gpu.py` | batched GPU simulator, Wiener model |
| `oracle_study/realdata/` | data preparation, AR(1) fit, real-data simulator and sweep |
| `oracle_study/report/` | LaTeX report, figure / table generation |
