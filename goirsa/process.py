from torch.nn.functional import conv2d, conv1d
from typing import List
import torch
import numpy as np

class Process:
    def __init__(self, 
                 num_nodes: int,
                 dim: int, # Dimension of the process (1 or 2)
                 n_bins: int,
                 max_value: float,
                 sigma: float,
                 device: torch.device = torch.device("cpu")):
        
        if dim not in [1, 2]:
            raise ValueError("Dimension must be either 1 or 2")

        self.num_nodes = num_nodes
        self.dim = dim
        self.n_bins = n_bins
        self.max_value = max_value
        self.sigma = sigma

        self.state = torch.zeros(num_nodes, dim) # N x D (D is either 1 or 2)

        # Define distribution 
        self.centers = torch.linspace(-max_value, max_value, n_bins + 1) # n_bins + 1 edges for n_bins bins
        self.points = self.centers # Default case for 1D
        self.belief = torch.zeros(num_nodes, n_bins +1) # N x n_bins, initialized to zero
        self.belief[:, n_bins//2] = 1.0 # Start with a belief concentrated at the center (0)
        if self.dim == 2: self.setup_2d_distribution()
    
        # A gaussian distribution centered at the origin with standard deviation sigma, evaluated at the points
        self.initial_distribution = torch.exp(-0.5 * (self.points / self.sigma)**2) 
        self.initial_distribution /= torch.sum(self.initial_distribution) # Normalize to sum to 1
        self.delta_distribution = torch.zeros_like(self.initial_distribution)
        if self.dim == 1:
            self.delta_distribution[n_bins//2 + 1] = 1.0 # A delta distribution that shifts the belief to the right by one bin
        else:
            self.delta_distribution[n_bins//2 + 1, n_bins//2 + 1] = 1.0 # A delta distribution that shifts the belief to the right by one bin in the x direction

        self.masks = torch.zeros(n_bins // 2 + 1, *([n_bins + 1] * self.dim)) # (n_bins//2 + 1) x (n_bins + 1) x (n_bins + 1) for the 2D case, (n_bins//2 + 1) x (n_bins + 1) for the 1D case
        for idx, distance in enumerate(self.centers[n_bins//2:]): # Only need to consider distances up to the maximum distance for the masks
            # find the points that are within the distance from the origin and set the corresponding mask to 1
            if self.dim == 1:
                mask = (torch.abs(self.points) <= distance).float() # (n_bins + 1,)
                self.masks[idx] = mask
            else:
                mask = (self.points <= distance).float() # (n_bins + 1) x (n_bins + 1)
                self.masks[idx] = mask

        self.device = device
        if device.type == "cuda":
            self.use_gpu()
            
    def setup_2d_distribution(self):
        # Create a 2D grid of points for the 2D case
        self.points = torch.zeros(self.n_bins +1, self.n_bins +1) # (n_bins + 1) x (n_bins + 1) grid of points
        for i in range(self.n_bins + 1):
            for j in range(self.n_bins + 1):
                self.points[i, j] = torch.sqrt(self.centers[i]**2 + self.centers[j]**2) # Distance from the origin

        self.belief = torch.zeros(self.num_nodes, self.n_bins +1, self.n_bins +1) # N x (n_bins + 1) x (n_bins + 1)
        self.belief[:, self.n_bins//2, self.n_bins//2] = 1.0 # Start with a belief concentrated at the center (0,0)

    def update_process(self):
        self.state += torch.randn_like(self.state) * self.sigma

    def beluef_update(self):
        # Belief shape is N, C
        if self.dim == 1:
            input_belief = self.belief.unsqueeze(1) # N x 1 x (n_bins + 1)
            kernel = self.initial_distribution.view(1, 1, -1) # 1 x 1 x (n_bins + 1)
            self.belief = conv1d(input_belief, kernel, padding=self.n_bins//2) # Convolve each node's belief with the kernel
            self.belief = self.belief.squeeze(1) # N x (n_bins + 1)
            sigma = torch.sqrt((self.belief * self.points**2).sum(dim=1)) # N
        else:
            input_belief = self.belief.unsqueeze(1) # N x 1 x (n_bins + 1) x (n_bins + 1)
            kernel = self.initial_distribution.view(1, 1, self.n_bins + 1, self.n_bins + 1) # 1 x 1 x (n_bins + 1) x (n_bins + 1)
            self.belief = conv2d(input_belief, kernel, padding=self.n_bins//2) # Convolve each node's belief with the kernel
            self.belief = self.belief.squeeze(1) # N x (n_bins + 1) x (n_bins + 1)

        # Normalize the belief to sum to 1 for each node
        self.belief /= torch.sum(self.belief, dim=list(range(1, self.belief.dim())), keepdim=True) 

    def update_public_info(self, observation, threshold_index):
        """
        Update the belief based on the observation
        observation is a tensor of shape (num_nodes,) with values in {0, 1, 2} 
        corresponding to the three possible outcomes of the IRSA frame:
        0: collision 
        1: no collision (successful decoding)
        2: successful decoding
        """
        # Find the succesfully decoded nodes and reset their distribution to the initial distribution
        decoded_nodes = (observation == 2).nonzero(as_tuple=True)[0]
        self.belief[decoded_nodes.to(self.device)] = self.delta_distribution
        self.state[decoded_nodes.to(self.device)] = 0.0 # Reset the state of the successfully decoded nodes to 0

        # Find the nodes that not collided and trucate their distribution
        no_collision_nodes = (observation == 1).nonzero(as_tuple=True)[0]
        self.belief[no_collision_nodes.to(self.device)] *= self.masks[threshold_index]

        # For the collided nodes don't do anything, as their belief remains unchanged

        # Update the belief based on the process dynamics (convolution with a gaussian kernel)
        self.beluef_update()

    def update(self, observation, threshold_index, index_of_decoded_nodes=None, first_replica_slot=None):
        self.update_public_info(observation, threshold_index)
        self.update_process()

    def reset(self):
        self.state = torch.zeros(self.num_nodes, self.dim, device=self.device)
        self.belief = torch.zeros(self.num_nodes, *([self.n_bins + 1] * self.dim), device=self.device)
        if self.dim == 1:
            self.belief[:, self.n_bins//2] = 1.0
        else:
            self.belief[:, self.n_bins//2, self.n_bins//2] = 1.0

    def use_gpu(self):
        self.state = self.state.cuda()
        self.belief = self.belief.cuda()
        self.centers = self.centers.cuda()
        self.points = self.points.cuda()
        self.initial_distribution = self.initial_distribution.cuda()
        self.masks = self.masks.cuda()
        self.delta_distribution = self.delta_distribution.cuda()

    def get_error(self):
        return torch.sqrt(torch.sum(self.state**2, dim=1)) # N,

    def get_error_distribution(self):
        # this should return a tensor of shape (num_nodes, n_bins//2 + 1) which 
        # represents the distribution of the error for each node based on the 
        # belief and the centers of the bins
        error_distribution = torch.zeros(self.num_nodes, self.n_bins//2 + 1, device=self.device) # N x (n_bins//2 + 1)
        for idx, mask in enumerate(self.masks):
            if self.dim == 1:
                error_distribution[:, idx] = torch.sum(self.belief * mask, dim=1) # Sum over the bins to get the probability of the error being within the distance corresponding to the mask
            else:
                error_distribution[:, idx] = torch.sum(self.belief * mask, dim=(1, 2)) # Sum over the bins to get the probability of the error being within the distance corresponding to the mask
        return error_distribution




class ProcessD:
    def __init__(self, 
                 num_nodes: int,
                 dim: int, # Dimension of the process (1 or 2)
                 n_bins: int,
                 max_value: float,
                 sigma: float,
                 device: torch.device = torch.device("cpu")):
        
        if dim not in [1, 2]:
            raise ValueError("Dimension must be either 1 or 2")

        self.num_nodes = num_nodes
        self.dim = dim
        self.n_bins = n_bins
        self.max_value = max_value
        self.sigma = sigma
        self.sigmas = torch.zeros(num_nodes, device=device) + sigma # N
        self.state = torch.zeros(num_nodes, dim, device=device) # N x D (D is either 1 or 2)
        # Add noise to sigmas 
        self.sigmas += (torch.rand(num_nodes, device=device) - 0.5) * sigma # Add noise to sigmas

        # Define distribution 
        self.centers = torch.linspace(-max_value, max_value, n_bins + 1) # n_bins + 1 edges for n_bins bins
        self.points = self.centers # Default case for 1D
        self.belief = torch.zeros(num_nodes, n_bins +1) # N x n_bins, initialized to zero
        self.belief[:, n_bins//2] = 1.0 # Start with a belief concentrated at the center (0)
        if self.dim == 2: self.setup_2d_distribution()
    
        # A gaussian distribution centered at the origin with standard deviation sigma, evaluated at the points
        self.initial_distribution = torch.exp(-0.5 * (self.points / self.sigma)**2) 
        self.initial_distribution /= torch.sum(self.initial_distribution) # Normalize to sum to 1
        self.delta_distribution = torch.zeros_like(self.initial_distribution)
        if self.dim == 1:
            self.delta_distribution[n_bins//2 + 1] = 1.0 # A delta distribution that shifts the belief to the right by one bin
        else:
            self.delta_distribution[n_bins//2 + 1, n_bins//2 + 1] = 1.0 # A delta distribution that shifts the belief to the right by one bin in the x direction

        self.masks = torch.zeros(n_bins // 2 + 1, *([n_bins + 1] * self.dim)) # (n_bins//2 + 1) x (n_bins + 1) x (n_bins + 1) for the 2D case, (n_bins//2 + 1) x (n_bins + 1) for the 1D case
        for idx, distance in enumerate(self.centers[n_bins//2:]): # Only need to consider distances up to the maximum distance for the masks
            # find the points that are within the distance from the origin and set the corresponding mask to 1
            if self.dim == 1:
                mask = (torch.abs(self.points) <= distance).float() # (n_bins + 1,)
                self.masks[idx] = mask
            else:
                mask = (self.points <= distance).float() # (n_bins + 1) x (n_bins + 1)
                self.masks[idx] = mask

        self.device = device
        if device.type == "cuda":
            self.use_gpu()
            
    def setup_2d_distribution(self):
        # Create a 2D grid of points for the 2D case
        self.points = torch.zeros(self.n_bins +1, self.n_bins +1) # (n_bins + 1) x (n_bins + 1) grid of points
        for i in range(self.n_bins + 1):
            for j in range(self.n_bins + 1):
                self.points[i, j] = torch.sqrt(self.centers[i]**2 + self.centers[j]**2) # Distance from the origin

        self.belief = torch.zeros(self.num_nodes, self.n_bins +1, self.n_bins +1) # N x (n_bins + 1) x (n_bins + 1)
        self.belief[:, self.n_bins//2, self.n_bins//2] = 1.0 # Start with a belief concentrated at the center (0,0)

    def update_process(self):
        self.state += torch.randn_like(self.state) * self.sigmas.unsqueeze(1) # N x D

    def beluef_update(self):
        # Belief shape is N, C
        if self.dim == 1:
            input_belief = self.belief.unsqueeze(1) # N x 1 x (n_bins + 1)
            kernel = self.initial_distribution.view(1, 1, -1) # 1 x 1 x (n_bins + 1)
            self.belief = conv1d(input_belief, kernel, padding=self.n_bins//2) # Convolve each node's belief with the kernel
            self.belief = self.belief.squeeze(1) # N x (n_bins + 1)
            sigma = torch.sqrt((self.belief * self.points**2).sum(dim=1)) # N
        else:
            input_belief = self.belief.unsqueeze(1) # N x 1 x (n_bins + 1) x (n_bins + 1)
            kernel = self.initial_distribution.view(1, 1, self.n_bins + 1, self.n_bins + 1) # 1 x 1 x (n_bins + 1) x (n_bins + 1)
            self.belief = conv2d(input_belief, kernel, padding=self.n_bins//2) # Convolve each node's belief with the kernel
            self.belief = self.belief.squeeze(1) # N x (n_bins + 1) x (n_bins + 1)

        # Normalize the belief to sum to 1 for each node
        self.belief /= torch.sum(self.belief, dim=list(range(1, self.belief.dim())), keepdim=True) 

    def update_public_info(self, observation, threshold_index):
        """
        Update the belief based on the observation
        observation is a tensor of shape (num_nodes,) with values in {0, 1, 2} 
        corresponding to the three possible outcomes of the IRSA frame:
        0: collision 
        1: no collision (successful decoding)
        2: successful decoding
        """
        # Find the succesfully decoded nodes and reset their distribution to the initial distribution
        decoded_nodes = (observation == 2).nonzero(as_tuple=True)[0]
        self.belief[decoded_nodes.to(self.device)] = self.delta_distribution
        self.state[decoded_nodes.to(self.device)] = 0.0 # Reset the state of the successfully decoded nodes to 0

        # Find the nodes that not collided and trucate their distribution
        no_collision_nodes = (observation == 1).nonzero(as_tuple=True)[0]
        self.belief[no_collision_nodes.to(self.device)] *= self.masks[threshold_index]

        # For the collided nodes don't do anything, as their belief remains unchanged

        # Update the belief based on the process dynamics (convolution with a gaussian kernel)
        self.beluef_update()

    def update(self, observation, threshold_index, index_of_decoded_nodes=None, first_replica_slot=None):
        self.update_public_info(observation, threshold_index)
        self.update_process()

    def reset(self):
        self.state = torch.zeros(self.num_nodes, self.dim, device=self.device)
        self.belief = torch.zeros(self.num_nodes, *([self.n_bins + 1] * self.dim), device=self.device)
        if self.dim == 1:
            self.belief[:, self.n_bins//2] = 1.0
        else:
            self.belief[:, self.n_bins//2, self.n_bins//2] = 1.0

    def use_gpu(self):
        self.state = self.state.cuda()
        self.belief = self.belief.cuda()
        self.centers = self.centers.cuda()
        self.points = self.points.cuda()
        self.initial_distribution = self.initial_distribution.cuda()
        self.masks = self.masks.cuda()
        self.delta_distribution = self.delta_distribution.cuda()

    def get_error(self):
        return torch.sqrt(torch.sum(self.state**2, dim=1)) # N,

    def get_error_distribution(self):
        # this should return a tensor of shape (num_nodes, n_bins//2 + 1) which 
        # represents the distribution of the error for each node based on the 
        # belief and the centers of the bins
        error_distribution = torch.zeros(self.num_nodes, self.n_bins//2 + 1, device=self.device) # N x (n_bins//2 + 1)
        for idx, mask in enumerate(self.masks):
            if self.dim == 1:
                error_distribution[:, idx] = torch.sum(self.belief * mask, dim=1) # Sum over the bins to get the probability of the error being within the distance corresponding to the mask
            else:
                error_distribution[:, idx] = torch.sum(self.belief * mask, dim=(1, 2)) # Sum over the bins to get the probability of the error being within the distance corresponding to the mask
        return error_distribution


class ProcessRTS:
    def __init__(self, 
                 num_nodes: int,
                 dim: int, # Dimension of the process (1 or 2)
                 n_bins: int,
                 max_value: float,
                 sigma: float,
                 frame_length: int,
                 device: torch.device = torch.device("cpu")):
        
        if dim not in [1, 2]:
            raise ValueError("Dimension must be either 1 or 2")

        self.num_nodes = num_nodes
        self.dim = dim
        self.n_bins = n_bins
        self.max_value = max_value
        self.sigma = sigma
        self.frame_length = frame_length

        self.state = torch.zeros(num_nodes, dim) # N x D (D is either 1 or 2)

        # Define distribution 
        self.centers = torch.linspace(-max_value, max_value, n_bins + 1) # n_bins + 1 edges for n_bins bins
        self.points = self.centers # Default case for 1D
        self.belief = torch.zeros(num_nodes, n_bins +1) # N x n_bins, initialized to zero
        self.belief[:, n_bins//2] = 1.0 # Start with a belief concentrated at the center (0)
        if self.dim == 2: self.setup_2d_distribution()
    
        # A gaussian distribution centered at the origin with standard deviation sigma, evaluated at the points
        self.initial_distribution = torch.exp(-0.5 * (self.points / self.sigma)**2) 
        self.initial_distribution /= torch.sum(self.initial_distribution) # Normalize to sum to 1

        # Since nodes sample their values in the same slot they communicate, the next sigma**2 should be scaled by the
        # time duration of the residual number of slots in the frame 
        # To be faster we precompute all the possible distributions 
        self.all_distributions = torch.stack([torch.exp(-0.5 * (self.points / (self.sigma*np.sqrt(t/self.frame_length)))**2) for t in range(self.frame_length, 0, -1)], dim=0) # frame_length x (n_bins + 1)
        self.all_distributions /= torch.sum(self.all_distributions, dim=1, keepdim=True) # Normalize to sum to 1
        self.all_sigmas = torch.tensor([self.sigma * np.sqrt(t/self.frame_length) for t in range(self.frame_length, 0, -1)], device=device) # frame_length


        # Move the distributions to the device
        self.all_distributions = self.all_distributions.to(device)
        self.all_sigmas = self.all_sigmas.to(device)

        self.delta_distribution = torch.zeros_like(self.initial_distribution)
        if self.dim == 1:
            self.delta_distribution[n_bins//2 + 1] = 1.0 # A delta distribution that shifts the belief to the right by one bin
        else:
            self.delta_distribution[n_bins//2 + 1, n_bins//2 + 1] = 1.0 # A delta distribution that shifts the belief to the right by one bin in the x direction

        self.masks = torch.zeros(n_bins // 2 + 1, *([n_bins + 1] * self.dim)) # (n_bins//2 + 1) x (n_bins + 1) x (n_bins + 1) for the 2D case, (n_bins//2 + 1) x (n_bins + 1) for the 1D case
        for idx, distance in enumerate(self.centers[n_bins//2:]): # Only need to consider distances up to the maximum distance for the masks
            # find the points that are within the distance from the origin and set the corresponding mask to 1
            if self.dim == 1:
                mask = (torch.abs(self.points) <= distance).float() # (n_bins + 1,)
                self.masks[idx] = mask
            else:
                mask = (self.points <= distance).float() # (n_bins + 1) x (n_bins + 1)
                self.masks[idx] = mask

        self.device = device
        if device.type == "cuda":
            self.use_gpu()
            
    def setup_2d_distribution(self):
        # Create a 2D grid of points for the 2D case
        self.points = torch.zeros(self.n_bins +1, self.n_bins +1) # (n_bins + 1) x (n_bins + 1) grid of points
        for i in range(self.n_bins + 1):
            for j in range(self.n_bins + 1):
                self.points[i, j] = torch.sqrt(self.centers[i]**2 + self.centers[j]**2) # Distance from the origin

        self.belief = torch.zeros(self.num_nodes, self.n_bins +1, self.n_bins +1) # N x (n_bins + 1) x (n_bins + 1)
        self.belief[:, self.n_bins//2, self.n_bins//2] = 1.0 # Start with a belief concentrated at the center (0,0)

    def update_process(self, index_of_decoded_nodes=None, first_replica_slot=None):
        # The not decoded nodes evolve according to the process dynamics, while the decoded nodes evolves according to
        # the sigma corresponding to the time duration remaining in the frame after their first replica slot
        if index_of_decoded_nodes is not None and first_replica_slot is not None:
            # Update the state of the decoded nodes with the corresponding sigma
            self.state[index_of_decoded_nodes] += torch.randn_like(self.state[index_of_decoded_nodes]) * self.all_sigmas[first_replica_slot - 1].unsqueeze(1)
            # Update the state of the not decoded nodes with the original sigma
            not_decoded_nodes = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
            not_decoded_nodes[index_of_decoded_nodes] = False
            self.state[not_decoded_nodes] += torch.randn_like(self.state[not_decoded_nodes]) * self.sigma
        else:
            self.state += torch.randn_like(self.state) * self.sigma
            
    def beluef_update(self, index_of_decoded_nodes=None, first_replica_slot=None):
        # Belief shape is N, C
        if self.dim == 1:
            input_belief = self.belief.unsqueeze(1) # N x 1 x (n_bins + 1)
            for first_slot in torch.unique(first_replica_slot):
                # Find the nodes that have their first replica in the same slot and convolve their belief with the corresponding distribution
                nodes_to_update = (first_replica_slot == first_slot).nonzero(as_tuple=True)[0]
                if len(nodes_to_update) == 0: continue

                kernel = self.all_distributions[first_slot - 1].view(1, 1, -1) # 1 x 1 x (n_bins + 1)
                input_belief[nodes_to_update] = conv1d(input_belief[nodes_to_update], kernel, padding=self.n_bins//2) # Convolve each node's belief with the kernel
            # All the other nodes that are not decoded should be convolved with the original distribution
            not_decoded_nodes = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
            if index_of_decoded_nodes is not None:
                not_decoded_nodes[index_of_decoded_nodes] = False
            input_belief[not_decoded_nodes] = conv1d(input_belief[not_decoded_nodes], self.initial_distribution.view(1, 1, -1), padding=self.n_bins//2)
            self.belief = input_belief.squeeze(1) # N x (n_bins + 1)

        else:
            input_belief = self.belief.unsqueeze(1) # N x 1 x (n_bins + 1) x (n_bins + 1)
            kernel = self.initial_distribution.view(1, 1, self.n_bins + 1, self.n_bins + 1) # 1 x 1 x (n_bins + 1) x (n_bins + 1)
            self.belief = conv2d(input_belief, kernel, padding=self.n_bins//2) # Convolve each node's belief with the kernel
            self.belief = self.belief.squeeze(1) # N x (n_bins + 1) x (n_bins + 1)

        # Normalize the belief to sum to 1 for each node
        self.belief /= torch.sum(self.belief, dim=list(range(1, self.belief.dim())), keepdim=True) 

    def update_public_info(self, observation, threshold_index, index_of_decoded_nodes=None, first_replica_slot=None):
        """
        Update the belief based on the observation
        observation is a tensor of shape (num_nodes,) with values in {0, 1, 2} 
        corresponding to the three possible outcomes of the IRSA frame:
        0: collision 
        1: no collision (successful decoding)
        2: successful decoding
        """
        # Find the succesfully decoded nodes and reset their distribution to the initial distribution
        decoded_nodes = (observation == 2).nonzero(as_tuple=True)[0]
        self.belief[decoded_nodes.to(self.device)] = self.delta_distribution
        self.state[decoded_nodes.to(self.device)] = 0.0 # Reset the state of the successfully decoded nodes to 0

        # Find the nodes that not collided and trucate their distribution
        no_collision_nodes = (observation == 1).nonzero(as_tuple=True)[0]
        self.belief[no_collision_nodes.to(self.device)] *= self.masks[threshold_index]

        # For the collided nodes don't do anything, as their belief remains unchanged

        # Update the belief based on the process dynamics (convolution with a gaussian kernel)
        self.beluef_update(index_of_decoded_nodes, first_replica_slot)

    def update(self, observation, threshold_index, index_of_decoded_nodes=None, first_replica_slot=None):
        self.update_public_info(observation, threshold_index, index_of_decoded_nodes, first_replica_slot)
        self.update_process(index_of_decoded_nodes, first_replica_slot)

    def reset(self):
        self.state = torch.zeros(self.num_nodes, self.dim, device=self.device)
        self.belief = torch.zeros(self.num_nodes, *([self.n_bins + 1] * self.dim), device=self.device)
        if self.dim == 1:
            self.belief[:, self.n_bins//2] = 1.0
        else:
            self.belief[:, self.n_bins//2, self.n_bins//2] = 1.0

    def use_gpu(self):
        self.state = self.state.cuda()
        self.belief = self.belief.cuda()
        self.centers = self.centers.cuda()
        self.points = self.points.cuda()
        self.initial_distribution = self.initial_distribution.cuda()
        self.masks = self.masks.cuda()
        self.delta_distribution = self.delta_distribution.cuda()

    def get_error(self):
        return torch.sqrt(torch.sum(self.state**2, dim=1)) # N,

    def get_error_distribution(self):
        # this should return a tensor of shape (num_nodes, n_bins//2 + 1) which 
        # represents the distribution of the error for each node based on the 
        # belief and the centers of the bins
        error_distribution = torch.zeros(self.num_nodes, self.n_bins//2 + 1, device=self.device) # N x (n_bins//2 + 1)
        for idx, mask in enumerate(self.masks):
            if self.dim == 1:
                error_distribution[:, idx] = torch.sum(self.belief * mask, dim=1) # Sum over the bins to get the probability of the error being within the distance corresponding to the mask
            else:
                error_distribution[:, idx] = torch.sum(self.belief * mask, dim=(1, 2)) # Sum over the bins to get the probability of the error being within the distance corresponding to the mask
        return error_distribution

