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