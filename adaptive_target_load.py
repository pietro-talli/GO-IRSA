import argparse
import torch
import numpy as np
from goirsa.pull_based import PullBased
from goirsa.irsa_target import TargetIRSA, TargetIRSAMultiThreshold
import pandas as pd
from goirsa.policy import ThresholdPolicy

parser = argparse.ArgumentParser(description="Run the adaptive target load policy and compare it " \
"with the pull-based policy.")
parser.add_argument("--nodes", type=int, default=2000, help="Number of nodes in the system")
parser.add_argument("--slot_per_frame", type=int, default=200, help="Number of slots per frame")
parser.add_argument("--dim", type=int, default=1, help="Dimension of the process (1 or 2)")
parser.add_argument("--n_bins", type=int, default=1000, help="Number of bins for the belief distribution")
parser.add_argument("--sigma", type=float, default=0.04, help="Standard deviation for the initial distribution")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--steps", type=int, default=200, help="Number of slots to simulate")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the simulation on (cuda or cpu)")
parser.add_argument("--target_loads", type=float, nargs="+", default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], help="List of target loads to test the TargetIRSA policy with")
parser.add_argument("--irsa_model", type=str, default="optimal", choices=["optimal", "imperfect"], help="IRSA model to use for the TargetIRSA policy (optimal or imperfect)")
parser.add_argument("--degree_dist", type=str, default="bcsa", help="Degree distribution to use for the TargetIRSA policy (bcsa or liva)")

parser.add_argument("--num_subcarriers", type=int, default=10, help="Number of subcarriers to use")

if __name__ == "__main__":
    args = parser.parse_args()

    args.max_value = 6

    import os
    # Unique folder name based on the the parameters
    folder_name = f"results/{args.degree_dist}/{args.irsa_model}/node_{args.nodes}_slot_{args.slot_per_frame}_sigma_{args.sigma}_dim_{args.dim}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the parameters in a json file
    import json
    with open(os.path.join(folder_name, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    frame_length = args.slot_per_frame // args.num_subcarriers

    device = torch.device(args.device)
    policy = ThresholdPolicy(args.nodes, 
                             args.dim, 
                             args.n_bins, 
                             args.max_value, 
                             args.sigma * np.sqrt(frame_length + 3), 
                             frame_length=frame_length,
                             replicas=[1, 2, 3], device=device)

    policy.reset()
    # Initialize the pull-based policy and test it
    if args.irsa_model == "imperfect":
        erasure = True
    else:        
        erasure = False
    pull_based = PullBased(policy.process, 
                           num_slot_per_frame=args.slot_per_frame, 
                           imperfect_channel=erasure,
                           num_subcarriers=args.num_subcarriers,
                           erasure_prob=0.0,
                           quantile_to_track=0.95)

    results = pull_based.test(args.episodes, args.steps)

    average_error = results["mean_reward"] / args.steps
    average_expected_error = results["mean_expected_reward"] / args.steps
    std_error = results["std_reward"] / args.steps
    std_expected_error = results["std_expected_reward"] / args.steps

    column_names = ["name", "average_reward", "std_reward", "throughput", "collision", "load", "quantile_reward", "std_quantile_reward"]
    results_df = pd.DataFrame(columns=column_names)

    new_row = pd.DataFrame({"name": ["PullBased"], 
                            "average_reward": [average_error], 
                            "std_reward": [std_error], 
                            "throughput": [1.0], 
                            "collision": [0.0], 
                            "load": [1.0],
                            "quantile_reward": results["mean_quantile_reward"] / args.steps,
                            "std_quantile_reward": results["std_quantile_reward"] / args.steps})
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    target_loads = args.target_loads


    # Run the TargetIRSA policy for each target load and save the results in a dataframe
    for target_load in target_loads:
        target_irsa = TargetIRSA(num_nodes=args.nodes, 
                                dim=args.dim, 
                                n_bins=args.n_bins, 
                                max_value=args.max_value, 
                                sigma=args.sigma * np.sqrt(frame_length + 3), # This makes it possible to compare different frame sizes
                                num_slot_per_frame=args.slot_per_frame,
                                target_load=target_load,
                                device=device,
                                irsa_model=args.irsa_model,
                                degree_dist=args.degree_dist,
                                num_subcarriers=args.num_subcarriers,
                                quantile_to_track=0.95)
        results = target_irsa.run(num_episodes=args.episodes, num_steps=args.steps)

        # save trace to a npy file
        np.save(os.path.join(folder_name, f"trace_target_irsa_{target_load}.npy"), results["trace"])

        reward = results["mean_reward"] / args.steps
        std_reward = results["std_reward"] / args.steps

        print(f"Average reward over {args.episodes} episodes: {reward:.4f}")
        for metric_name, metric_value in results["metrics"].items():
            print(f"Average {metric_name} over {args.episodes} episodes: {np.mean(metric_value):.4f}")

        new_row = pd.DataFrame({"name": [f"TargetIRSA_{target_load}"], 
                                "average_reward": [reward], 
                                "std_reward": [std_reward],
                                "throughput": [np.mean(results["metrics"]["throughput"])], 
                                "collision": [np.mean(results["metrics"]["collision"])], 
                                "load": [np.mean(results["metrics"]["load"])],
                                "quantile_reward": [results["mean_quantile_reward"] / args.steps],
                                "std_quantile_reward": [results["std_quantile_reward"] / args.steps]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Run the TargetIRSAMultiThreshold policy and save the results in a dataframe
    '''
    for target_load in target_loads:
        target_irsa_mt = TargetIRSAMultiThreshold(num_nodes=args.nodes, 
                                dim=args.dim, 
                                n_bins=args.n_bins, 
                                max_value=args.max_value, 
                                sigma=args.sigma * np.sqrt(frame_length + 3), # This makes it possible to compare different frame sizes
                                num_slot_per_frame=args.slot_per_frame,
                                target_load=target_load,
                                device=device,
                                irsa_model=args.irsa_model,
                                degree_dist=args.degree_dist,
                                num_subcarriers=args.num_subcarriers,
                                quantile_to_track=0.9)
        results = target_irsa_mt.run(num_episodes=args.episodes, num_steps=args.steps)

        # save trace to a npy file
        np.save(os.path.join(folder_name, f"trace_target_irsa_mt_{target_load}.npy"), results["trace"])
        # save collision probability per replica to a json file
        with open(os.path.join(folder_name, f"collision_prob_per_replica_target_irsa_mt_{target_load}.json"), "w") as f:
            json.dump({target_irsa_mt.replicas[i]: results["cppr"][i] for i in range(len(target_irsa_mt.replicas))}, f, indent=4)

        reward = results["mean_reward"] / args.steps
        std_reward = results["std_reward"] / args.steps
        print(f"Average reward over {args.episodes} episodes: {reward:.4f}")
        for metric_name, metric_value in results["metrics"].items():
            print(f"Average {metric_name} over {args.episodes} episodes: {np.mean(metric_value):.4f}")

        new_row = pd.DataFrame({"name": [f"IRSA_MT_{target_load}"], 
                                "average_reward": [reward], 
                                "std_reward": [std_reward],
                                "throughput": [np.mean(results["metrics"]["throughput"])], 
                                "collision": [np.mean(results["metrics"]["collision"])], 
                                "load": [np.mean(results["metrics"]["load"])],
                                "quantile_reward": [results["mean_quantile_reward"] / args.steps],
                                "std_quantile_reward": [results["std_quantile_reward"] / args.steps]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    '''
    results_df.to_csv(os.path.join(folder_name, "results.csv"), index=False)

    # Create a tensorboard file for the results
    from torch.utils.tensorboard import SummaryWriter   
    writer = SummaryWriter(os.path.join(folder_name, "tensorboard/PULL"))
    writer.add_scalar("metrics/perf", average_error, 0)
    writer.add_scalar("metrics/perf", average_error, 100)
    writer.close()

    # Create a tensorboard file for the results of the TargetIRSA policy
    writer = SummaryWriter(os.path.join(folder_name, "tensorboard/TARGET_IRSA"))
    # Find the rows corresponding to the TargetIRSA policy and plot the average reward as a function of the target load
    for name in results_df["name"].unique():
        if "TargetIRSA" in name:
            reward = results_df[results_df["name"] == name]["average_reward"].values[0]
            load = results_df[results_df["name"] == name]["load"].values[0]
            writer.add_scalar("metrics/perf", reward, int(load*100))
    writer.close()
    
    # Create a tensorboard file for the results of the IRSA_MT policy
    '''
    writer = SummaryWriter(os.path.join(folder_name, "tensorboard/IRSA_MT"))
    # Find the rows corresponding to the IRSA_MT policy and plot the average reward as a function of the target load
    for name in results_df["name"].unique():
        if "IRSA_MT" in name:
            reward = results_df[results_df["name"] == name]["average_reward"].values[0]
            load = results_df[results_df["name"] == name]["load"].values[0]
            writer.add_scalar("metrics/perf", reward, int(load*100))
    writer.close()
    '''