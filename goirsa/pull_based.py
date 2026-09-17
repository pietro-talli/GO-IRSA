from goirsa.process import Process
import torch
import numpy as np

class PullBased:
    def __init__(self, process: Process,
                 num_slot_per_frame: int,
                 imperfect_channel: bool = False,
                 num_subcarriers: int = 1,
                 erasure_prob: float = 5e-3,
                 quantile_to_track: float = 0.95):
        self.process = process
        self.num_slot_per_frame = num_slot_per_frame
        self.imperfect_channel = imperfect_channel
        self.num_subcarriers = num_subcarriers
        self.quantile_to_track = quantile_to_track
        if self.imperfect_channel:
            self.erasure_prob = erasure_prob

        self.selected_nodes_couter = torch.zeros(self.process.num_nodes, device=self.process.device)
        # Num nodes 
        self.num_nodes = self.process.num_nodes

        self.frame_length = self.num_slot_per_frame // self.num_subcarriers

    def reset(self):
        self.process.reset()

    def step(self, selected_nodes):
        # Select the nodes to transmit in the frame based on the erro distribution 
        error_distribution = self.process.get_error_distribution() # N x num_bins

        # finde the num_slot_per_frame nodes with the highest error distribution
        # Get the pdf of the error distribution from the cdf 
        pdf = error_distribution[:, 1:] - error_distribution[:, :-1] # N x (num_bins - 1)

        # Get the indices of the nodes with the expected error 
        expected_error = (pdf * torch.arange(1, pdf.shape[1] + 1, device=self.process.device)).sum(dim=1) # N
        _, selected_nodes = torch.topk(expected_error, self.num_slot_per_frame) # num_slot_per_frame
        observation = torch.zeros(self.process.num_nodes, dtype=torch.long, device=self.process.device) # N

        self.selected_nodes_couter[selected_nodes] += 1

        if self.imperfect_channel:
            # Simulate erasures
            erasures = torch.rand(len(selected_nodes), device=self.process.device) < self.erasure_prob
            selected_nodes = selected_nodes[~erasures] # Only keep the nodes that are not erased


        index_of_decoded_nodes = selected_nodes
        first_replica_slot = torch.arange(len(selected_nodes), device=self.process.device) % self.frame_length + 1 # Assuming the first replica is transmitted in the slot corresponding to the node index modulo the frame length

        observation[selected_nodes] = 2 # Mark the selected nodes
        self.process.update(observation, threshold_index=0, index_of_decoded_nodes=index_of_decoded_nodes, first_replica_slot=first_replica_slot)

        # compute the expected reward
        expected_error = self.process.belief * self.process.points**2
        if self.process.dim == 1:
            expected_error = expected_error.sum(dim=1) # N
        else:
            expected_error = expected_error.sum(dim=(1, 2)) # N
        expected_reward = -torch.mean(expected_error).item()
        # not_selected_nodes = torch.ones(self.process.num_nodes, dtype=torch.bool, device=self.process.device)
        # not_selected_nodes[selected_nodes] = False
        # r = -(self.process.get_error()[not_selected_nodes]**2).sum()
        # r -= (self.process.get_error()[selected_nodes]**2).sum() * (((self.frame_length + 3) +1)/(2*(self.frame_length+3)))
        # reward = (r/self.num_nodes).item()
        reward = -torch.mean(self.process.get_error()**2).item()
        quantile_reward = -torch.quantile(self.process.get_error()**2, self.quantile_to_track).item()
        
        return reward, expected_reward, quantile_reward
    
    def test(self, num_epidoes: int, num_steps: int):
        rewards = np.zeros(num_epidoes*num_steps)
        expected_rewards = np.zeros(num_epidoes*num_steps)
        quantile_rewards = np.zeros(num_epidoes*num_steps)
        for episode in range(num_epidoes):
            self.reset()
            episode_reward = 0
            episode_expected_reward = 0
            episode_quantile_reward = 0
            for step in range(num_steps):
                selected_nodes = np.arange(step * self.num_slot_per_frame, (step + 1) * self.num_slot_per_frame) % self.process.num_nodes
                reward, expected_reward, quantile_reward = self.step(torch.tensor(selected_nodes, device=self.process.device))
                episode_reward += reward
                episode_expected_reward += expected_reward
                episode_quantile_reward += quantile_reward
            rewards[episode * num_steps:(episode + 1) * num_steps] = episode_reward
            expected_rewards[episode * num_steps:(episode + 1) * num_steps] = episode_expected_reward
            quantile_rewards[episode * num_steps:(episode + 1) * num_steps] = episode_quantile_reward


        # print(self.selected_nodes_couter)
        self.selected_nodes_couter = torch.zeros(self.process.num_nodes, device=self.process.device)
        return {
            "mean_reward": np.mean(rewards),
            "mean_expected_reward": np.mean(expected_rewards),
            "std_reward": np.std(rewards),
            "std_expected_reward": np.std(expected_rewards),
            "mean_quantile_reward": np.mean(quantile_rewards),
            "std_quantile_reward": np.std(quantile_rewards)
        }