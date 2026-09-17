import torch
from goirsa.process import Process, ProcessD, ProcessRTS

class Policy:
    def __init__(self, num_nodes, dim, n_bins, max_value, sigma, frame_length, device):
        # self.process = ProcessRTS(num_nodes, dim, n_bins, max_value, sigma, frame_length, device)
        # self.process = Process(num_nodes, dim, n_bins, max_value, sigma, device)
        self.process = ProcessD(num_nodes, dim, n_bins, max_value, sigma, device)

    def update(self, observation, threshold_index, index_of_decoded_nodes = None, first_replica_slot = None):
        # Update the process with the new observations
        self.process.update(observation, threshold_index, index_of_decoded_nodes, first_replica_slot)

    def reset(self):
        self.process.reset()

    def get_action(self):
        raise NotImplementedError("This method should be implemented by subclasses to return the action based on the current belief and state of the process.")
    
    def get_error_distribution(self):
        return self.process.get_error_distribution()

from typing import List 

class ThresholdPolicy(Policy):
    def __init__(self, num_nodes, dim, n_bins, max_value, sigma, replicas: List[int], 
                 frame_length: int,
                 device = torch.device("cpu")):
        super().__init__(num_nodes, dim, n_bins, max_value, sigma, frame_length, device)
        self.replicas = replicas
        self.n_bins = n_bins

    def get_action(self, thresholds):
        # Compare the private information with the threshold to decide the number of replicas to transmit
        error = self.process.get_error()
        action = torch.zeros_like(error, dtype=torch.long)
        for idx, num_replicas in enumerate(self.replicas):
            action[error > self.process.centers[self.n_bins // 2 + thresholds[idx]]] = num_replicas
        return action
    
    def get_action_from_continuous(self, thresholds):
        # Compare the private information with the threshold to decide the number of replicas to transmit
        error = self.process.get_error()
        action = torch.zeros_like(error, dtype=torch.long)
        for idx, num_replicas in enumerate(self.replicas):
            action[error > thresholds[idx]] = num_replicas
        return action
