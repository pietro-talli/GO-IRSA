import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from goirsa.process import Process
from goirsa.policy import ThresholdPolicy

from goirsa.irsa_util import imperfect_irsa, optimal_irsa, degree_distribution, irsa_fast

class IRSAEnv(gym.Env):
    def __init__(self, 
                 num_nodes: int,
                 dim: int, # Dimension of the process (1 or 2)
                 n_bins: int,
                 max_value: float,
                 sigma: float,
                 num_slot_per_frame: int,
                 num_steps: int,
                 replicas: list,
                 device: torch.device = torch.device("cpu"),
                 irsa_model: str = "optimal",
                 num_subcarriers: int = 1,
                 num_envs: int = 1):
        
        self.frame_length = num_slot_per_frame // num_subcarriers
        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma, 
                                      frame_length=self.frame_length, replicas=replicas, device=device)
        self.num_slot_per_frame = num_slot_per_frame
        self.num_steps = num_steps

        self.num_nodes = num_nodes
        self.replicas = replicas
        self.irsa_model = irsa_model
        if irsa_model == "optimal":
            self.irsa_func = irsa_fast
        elif irsa_model == "imperfect":
            self.irsa_func = imperfect_irsa

        # Define action and observation spaces
        num_thresholds = len(replicas)
        self.n_bins = n_bins
        self.action_space = spaces.MultiDiscrete(n_bins // 2 * np.ones(num_thresholds, dtype=int))
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes, n_bins//2 + 1), dtype=np.float32) # Error distribution for each node

        self.time_step = 0

        self.num_subcarriers = num_subcarriers
        
        # If num_envs > 1, force the process use os.cpucount() // num_envs cores to avoid over-subscription
        if num_envs > 1:
            torch.set_num_threads(1) # torch.get_num_threads() // num_envs)
            # Allow numpy to use only a fraction of the cores to avoid over-subscription
            import os
            # os.environ["OMP_NUM_THREADS"] = str(1) #os.cpu_count() // num_envs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.time_step = 0
        self.policy.reset()
        return self.policy.get_error_distribution().cpu().numpy(), {}

    def step(self, action):
        # Rewrite actions as the cumulative sum of the thresholds
        action = np.cumsum(action)
        # Clip the action to be within the valid range
        action = np.clip(action, 0, self.n_bins // 2)
        # action are the threshold indices
        num_replicas = self.policy.get_action(thresholds=action)
        slots, first_replica_slot = self.select_slots(num_replicas)
        slots = slots.cpu().numpy()
        obs = self.simulate_frame(slots)
        index_of_decoded_nodes = (obs == 2).nonzero()[0]
        self.policy.update(observation=torch.from_numpy(obs).long(), 
                   threshold_index=action[0],
                   index_of_decoded_nodes=index_of_decoded_nodes,
                   first_replica_slot=first_replica_slot[index_of_decoded_nodes])
        
        # index_of_not_decoded_nodes = (obs != 2).nonzero()[0]
        # r = -(self.policy.process.get_error()[index_of_not_decoded_nodes]**2).sum()
        # if len(index_of_decoded_nodes) > 0:
        #     r -= (self.policy.process.get_error()[index_of_decoded_nodes]**2).sum() * ((((self.frame_length + 3 - first_replica_slot[index_of_decoded_nodes]).mean().item())+1)/(2*(self.frame_length + 3)))
        # reward = (r/self.num_nodes).item()
        reward = -np.mean(self.policy.process.get_error().cpu().numpy()**2) # Negative of the average error across nodes
        next_state = self.policy.get_error_distribution().cpu().numpy()

        self.time_step += 1
        truncated = self.time_step >= self.num_steps

        # Some IRSA metrics
        self.tx_attempts = np.sum(num_replicas.cpu().numpy() > 0)
        self.successful_decodings = np.sum(obs == 2)
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        info = {
            "throughput": self.throughput,
            "collision": self.collision,
            "load": self.load,
        }

        return next_state, reward, False, truncated, info


    def select_slots(self, num_replicas):
        # Each nodes selects transmitting slots based on the num_replicas selected by the policy
        first_replica_slot = torch.zeros(self.num_nodes, dtype=torch.long, device=num_replicas.device)
        slots = torch.zeros(self.num_nodes, self.num_slot_per_frame)
        for i in range(self.num_nodes):
            if num_replicas[i] > 0:
                if self.num_subcarriers > 1:
                    # print(self.frame_length, num_replicas[i])
                    selected_slots = torch.randint(0, self.frame_length, (num_replicas[i],), device=slots.device)
                    selected_carriers = torch.randint(0, self.num_subcarriers, (num_replicas[i],), device=slots.device)
                    slots[i, selected_slots + selected_carriers*self.frame_length] = 1
                    first_replica_slot[i] = selected_slots[0]
                else:
                    selected_slots = torch.randint(0, self.num_slot_per_frame, (num_replicas[i],), device=slots.device)
                    slots[i, selected_slots] = 1
                    first_replica_slot[i] = selected_slots[0]    
        return slots, first_replica_slot
    
    def simulate_frame(self, slot_selection):
        return self.irsa_func(slot_selection)
    


class IRSAEnvThreshold(gym.Env):
    def __init__(self, 
                 num_nodes: int,
                 dim: int, # Dimension of the process (1 or 2)
                 n_bins: int,
                 max_value: float,
                 sigma: float,
                 num_slot_per_frame: int,
                 num_steps: int,
                 replicas: list,
                 device: torch.device = torch.device("cpu"),
                 irsa_model: str = "optimal",
                 num_envs: int = 1,
                 num_subcarriers: int = 1,
                 degree_dist: str = "bcsa"):
        
        self.frame_length = num_slot_per_frame // num_subcarriers
        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma, 
                                      frame_length=self.frame_length, replicas=replicas, device=device)
        self.num_slot_per_frame = num_slot_per_frame
        self.num_steps = num_steps
        self.degree_dist = degree_dist

        self.num_nodes = num_nodes
        self.replicas = replicas
        self.irsa_model = irsa_model
        if irsa_model == "optimal":
            self.irsa_func = irsa_fast
        elif irsa_model == "imperfect":
            self.irsa_func = imperfect_irsa

            
        self.replicas = degree_distribution[self.degree_dist]["replicas"]
        self.replicas_prob = degree_distribution[self.degree_dist]["replicas_prob"]

        # Define action and observation spaces
        self.n_bins = n_bins
        self.action_space = spaces.Discrete(n_bins // 2) # Action is the index of the threshold
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes, n_bins//2 + 1), dtype=np.float32) # Error distribution for each node

        self.time_step = 0

        # If num_envs > 1, force the process use os.cpucount() // num_envs cores to avoid over-subscription
        if num_envs > 1:
            torch.set_num_threads(1) # torch.get_num_threads() // num_envs)
            # Allow numpy to use only a fraction of the cores to avoid over-subscription
            import os
            os.environ["OMP_NUM_THREADS"] = str(1) #os.cpu_count() // num_envs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.time_step = 0
        self.policy.reset()
        return self.policy.get_error_distribution().cpu().numpy(), {}


    def select_replicas(self, threshold_index):
        # Find the nodes whose error is above the threshold
        nodes_above_threshold = (self.policy.process.get_error() > self.policy.process.centers[self.policy.process.n_bins//2 + threshold_index]).nonzero(as_tuple=True)[0]
        num_nodes_above_threshold = len(nodes_above_threshold)
        if num_nodes_above_threshold == 0:
            return torch.zeros(self.num_nodes, dtype=torch.long, device=self.policy.process.device)
        # Select the replicas for the nodes above the threshold according to the degree distribution
        selected_replicas = np.random.choice(self.replicas, size=num_nodes_above_threshold, p=self.replicas_prob)
        replicas_tensor = torch.zeros(self.num_nodes, dtype=torch.long, device=self.policy.process.device)
        replicas_tensor[nodes_above_threshold] = torch.from_numpy(selected_replicas).to(self.policy.process.device)
        return replicas_tensor
    
    def select_slots(self, replicas):
        # Select the slots for the nodes based on their replicas
        replicas = replicas.cpu().numpy()
        slot_selection = np.zeros((self.num_nodes, self.num_slot_per_frame))
        for i in range(self.num_nodes):
            if replicas[i] > 0:
                selected_slots = np.random.choice(self.num_slot_per_frame, size=replicas[i], replace=False)
                slot_selection[i, selected_slots] = 1
        return slot_selection


    def step(self, action):
        # action is the threshold index
        replicas = self.select_replicas(action)
        slot_selection = self.select_slots(replicas)
        obs = self.simulate_frame(slot_selection)
        self.policy.update(observation=torch.from_numpy(obs).long(), threshold_index=action)
        reward = -np.mean(self.policy.process.get_error().cpu().numpy()**2) # Negative of the average error across nodes
        next_state = self.policy.get_error_distribution().cpu().numpy()

        self.time_step += 1
        truncated = self.time_step >= self.num_steps

        # Some IRSA metrics
        self.tx_attempts = np.sum(replicas.cpu().numpy() > 0)
        self.successful_decodings = np.sum(obs == 2)
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        info = {
            "throughput": self.throughput,
            "collision": self.collision,
            "load": self.load,
        }

        return next_state, reward, False, truncated, info
    
    def simulate_frame(self, slot_selection):
        return self.irsa_func(slot_selection)
    