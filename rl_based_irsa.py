import argparse
import json
import os
import torch
import numpy as np
from goirsa.environment import IRSAEnv, IRSAEnvThreshold # Una threshold intanto 
import pandas as pd

from rl_model.feature_extractor import ConvolutionalFeatureExtractor, LinearLoadFeatureExtractor
from rl_model.rl_utils import CustomTensorboardCallback, evaluate_policy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from goirsa.irsa_util import degree_distribution

parser = argparse.ArgumentParser(description="Run the RL-based policy for IRSA")
parser.add_argument("--nodes", type=int, default=2000, help="Number of nodes in the system")
parser.add_argument("--slot_per_frame", type=int, default=400, help="Number of slots per frame")
parser.add_argument("--dim", type=int, default=1, help="Dimension of the process (1 or 2)")
parser.add_argument("--n_bins", type=int, default=50, help="Number of bins for the belief distribution")
parser.add_argument("--sigma", type=float, default=0.04, help="Standard deviation for the initial distribution")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
parser.add_argument("--steps", type=int, default=200, help="Total number of slots to simulate")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the simulation on (cuda or cpu)")
parser.add_argument("--irsa_model", type=str, default="optimal", choices=["optimal", "imperfect"], help="IRSA model to use for the TargetIRSA policy (optimal or imperfect)")
parser.add_argument("--degree_dist", type=str, default="bcsa", help="Degree distribution to use for the TargetIRSA policy (bcsa or liva)")

parser.add_argument("--num_subcarriers", type=int, default=10, help="Number of subcarriers to use")

# Params for RL training
parser.add_argument("--training_steps", type=int, default=1000000, help="Number of training steps for the RL agent")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments for training")

if __name__ == "__main__":
    args = parser.parse_args()

    args.max_value = 6

    device = torch.device(args.device)
    folder_name = f"results/{args.degree_dist}/{args.irsa_model}/node_{args.nodes}_slot_{args.slot_per_frame}_sigma_{args.sigma}_dim_{args.dim}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the arguments to a json file for future reference
    args_file = os.path.join(folder_name, "rl_args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4)

    # If exist load the existing results.csv file, otherwise create a new one
    results_file = os.path.join(folder_name, "results.csv")
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        column_names = ["name", "average_reward", "std_reward", "throughput", "collision", "load"]
        results_df = pd.DataFrame(columns=column_names)

    replicas = degree_distribution[args.degree_dist]["replicas"]

    frame_length = args.slot_per_frame // args.num_subcarriers

    env_fns = [lambda: Monitor(IRSAEnvThreshold(num_nodes=args.nodes,
                               dim=args.dim,
                               n_bins=args.n_bins,
                               max_value=args.max_value,
                               sigma=args.sigma * np.sqrt(frame_length + 3), # This makes it possible to compare different frame sizes
                               num_slot_per_frame=args.slot_per_frame,
                               num_steps=args.steps,
                               replicas=replicas,
                               device=device,
                               irsa_model=args.irsa_model,
                               num_subcarriers=args.num_subcarriers,
                               num_envs=args.num_envs,
                               degree_dist=args.degree_dist)) for _ in range(args.num_envs)]
    env = SubprocVecEnv(env_fns, start_method="spawn")

    model = PPO("MlpPolicy", 
                env,
                gamma=0.99,
                n_steps = 5000 // args.num_envs,
                batch_size = 625,
                learning_rate=3e-4,
                ent_coef=0.1,
                clip_range=0.2,
                vf_coef=0.4,
                policy_kwargs={"features_extractor_class": ConvolutionalFeatureExtractor, 
                               "net_arch": [512, 256, 256],
                               "optimizer_class": torch.optim.AdamW,
                               "optimizer_kwargs": {"weight_decay": 1e-5}
                               },
                tensorboard_log=os.path.join(folder_name, "tensorboard"),
                verbose=0)

    callback = CustomTensorboardCallback()

    for round in range(0, args.training_steps // 10000):
        if round > 0:
            model.learn(total_timesteps=10000, callback=callback, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=10000, callback=callback)

        # Evaluate the trained model
        reward, std, load = evaluate_policy(model, env, args.num_envs, n_time_steps=args.steps*args.num_envs)
        reward = reward / args.steps
        std = std / args.steps
        print(f"Average Reward: {reward:.2f}, Std Reward: {std:.2f}, Load: {load:.2f}")

        model.logger.output_formats[-1].writer.add_scalar("metrics/perf", reward, int(load * 100))


    model.save(os.path.join(folder_name, "ppo_irsa_model"))

    