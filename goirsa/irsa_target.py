from goirsa.policy import ThresholdPolicy
import torch
import numpy as np
from goirsa.irsa_util import optimal_irsa, imperfect_irsa, degree_distribution, optimal_irsa_fast, irsa_fast
import time

class TargetIRSA:
    def __init__(self, 
                 num_nodes: int, 
                 dim: int, 
                 n_bins: int, 
                 max_value: float, 
                 sigma: float, 
                 num_slot_per_frame: int,
                 target_load: float,
                 device: torch.device,
                 irsa_model: str = "optimal",
                 degree_dist: str = "bcsa",
                 num_subcarriers: int = 1,
                 quantile_to_track: float = 0.95):
        
        self.frame_length = num_slot_per_frame // num_subcarriers

        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma, replicas=[1], 
                                      frame_length=self.frame_length, device=device)
        self.quantile_to_track = quantile_to_track
        self.num_nodes = num_nodes
        self.num_slot_per_frame = num_slot_per_frame
        self.target_load = target_load
        self.irsa_model_name = irsa_model
        self.num_subcarriers = num_subcarriers
        
        if irsa_model == "optimal":
            self.irsa_model = optimal_irsa
        else:
            self.irsa_model = imperfect_irsa
        self.degree_dist = degree_dist

        self.collision = 0.0
        self.threshold_index = 0
        

        # This degree distribution is found in https://ieeexplore.ieee.org/document/7736044
        self.replicas = degree_distribution[self.degree_dist]["replicas"]
        self.replicas_prob = degree_distribution[self.degree_dist]["replicas_prob"]

    def reset(self):
        self.policy.reset()

    def select_threshold(self, error_distribution):
        # Select the threshold such that the expected load is equal to the target load
        expected_silent_nodes = error_distribution.sum(dim=0) # num_bins
        expected_load = (self.num_nodes - expected_silent_nodes) / self.num_slot_per_frame 
        # Find the threshold index such that the closest expected load to the target load is achieved
        threshold_index = torch.argmin(torch.abs(expected_load - self.target_load)).item()

        # variance = (error_distribution[:,threshold_index] * (1 - error_distribution[:,threshold_index])).sum().item()

        real_load = (self.policy.process.get_error() > self.policy.process.centers[self.policy.process.n_bins//2 + threshold_index]).sum().item() / self.num_slot_per_frame
        # bias = real_load - expected_load[threshold_index].item()
        self.threshold_index = threshold_index
        return threshold_index 
    
    def select_replicas(self, threshold_index):
        # print(self.policy.process.centers[self.policy.process.n_bins//2 + threshold_index])
        #p = 0.5*self.num_slot_per_frame/self.num_nodes


        # Find the nodes whose error is above the threshold
        nodes_above_threshold = (self.policy.process.get_error() > self.policy.process.centers[self.policy.process.n_bins//2 + threshold_index]).nonzero(as_tuple=True)[0]

        # Sample nodes with prob p
        #nodes_above_threshold = torch.bernoulli(torch.full((self.num_nodes,), p, device=self.policy.process.device)).nonzero(as_tuple=True)[0]

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
        first_replica_slot = torch.zeros(self.num_nodes, dtype=torch.long, device=self.policy.process.device)
        slot_selection = np.zeros((self.num_nodes, self.num_slot_per_frame))
        for i in range(self.num_nodes):
            if replicas[i] > 0:
                if self.num_subcarriers > 1:
                    selected_slots = np.random.choice(self.frame_length, size=replicas[i], replace=False)
                    selected_carriers = np.random.choice(self.num_subcarriers, size=replicas[i], replace=True)
                    slot_selection[i, selected_slots + selected_carriers*self.frame_length] = 1
                    first_replica_slot[i] = selected_slots[0]
                else:
                    selected_slots = np.random.choice(self.num_slot_per_frame, size=replicas[i], replace=False)
                    slot_selection[i, selected_slots] = 1
                    first_replica_slot[i] = selected_slots[0]
        return slot_selection, first_replica_slot

    def simulate_frame(self, slot_selection):
        return self.irsa_model(slot_selection)
    
    def step(self):
        error_distribution = self.policy.get_error_distribution()
        threshold_index = self.select_threshold(error_distribution)
        replicas = self.select_replicas(threshold_index)
        slot_selection, first_replica_slot = self.select_slots(replicas)
        observation = self.simulate_frame(slot_selection)
        # Some IRSA metrics
        self.tx_attempts = np.sum(replicas.cpu().numpy() > 0)
        self.successful_decodings = np.sum(observation == 2)
        index_of_decoded_nodes = (observation == 2).nonzero()[0]
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        self.policy.update(observation=torch.from_numpy(observation).to(self.policy.process.device), 
                           threshold_index=threshold_index, 
                           index_of_decoded_nodes=index_of_decoded_nodes,
                           first_replica_slot=first_replica_slot[index_of_decoded_nodes])
        return index_of_decoded_nodes, first_replica_slot[index_of_decoded_nodes]
    
    def run(self, num_episodes: int, num_steps: int):
        with torch.no_grad():
            rewards = np.zeros(num_episodes*num_steps)
            quantile_rewards = np.zeros(num_episodes*num_steps)

            metrics = {
                "throughput": np.zeros(num_episodes*num_steps),
                "collision": np.zeros(num_episodes*num_steps),
                "load": np.zeros(num_episodes*num_steps)
            }

            trace = np.zeros((num_episodes, num_steps, 2)) # (num_episodes, num_steps, [th, collision])

            for episode in range(num_episodes):
                self.reset()
                episode_reward = 0
                episode_quantile_reward = 0
                for step in range(num_steps):   
                    #time_start = time.time()         
                    index_of_decoded_nodes, first_replica_slot = self.step() 
                    #time_end = time.time()
                    #print(f"Step time: {time_end - time_start} seconds")    
                    # # All the other nodes
                    # not_selected_nodes = torch.ones(self.policy.process.num_nodes, dtype=torch.bool, device=self.policy.process.device)
                    # not_selected_nodes[index_of_decoded_nodes] = False
                    # r = -(self.policy.process.get_error()[not_selected_nodes]**2).sum()
                    # if len(index_of_decoded_nodes) > 0:
                    #     r -= (self.policy.process.get_error()[index_of_decoded_nodes]**2).sum() * (((((self.frame_length + 3) - first_replica_slot).mean().item())+1)/(2*(self.frame_length+3)))
                    # episode_reward += (r/self.num_nodes).item()
                    error = self.policy.process.get_error()**2
                    reward = -torch.mean(error).item()
                    episode_reward += reward
                    quantile_reward = -torch.quantile(error, self.quantile_to_track).item()
                    episode_quantile_reward += quantile_reward
                    metrics["throughput"][episode*num_steps + step] = self.throughput
                    metrics["collision"][episode*num_steps + step] = self.collision
                    metrics["load"][episode*num_steps + step] = self.load

                    trace[episode, step, 0] = self.threshold_index
                    trace[episode, step, 1] = self.collision

                    # print(torch.max(self.policy.process.get_error()).item())

                rewards[episode*num_steps:(episode+1)*num_steps] = episode_reward
                quantile_rewards[episode*num_steps:(episode+1)*num_steps] = episode_quantile_reward
                # print(max(self.policy.process.get_error().cpu().numpy()), metrics["load"][-1])
            return {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "mean_quantile_reward": np.mean(quantile_rewards),
                "std_quantile_reward": np.std(quantile_rewards),
                "metrics": metrics,
                "trace": trace
            }
    


class TargetIRSAMultiThreshold:
    def __init__(self, 
                 num_nodes: int, 
                 dim: int, 
                 n_bins: int, 
                 max_value: float, 
                 sigma: float, 
                 num_slot_per_frame: int,
                 target_load: float,
                 device: torch.device,
                 irsa_model: str = "optimal",
                 degree_dist: str = "bcsa",
                 num_subcarriers: int = 1,
                 quantile_to_track: float = 0.95):
        self.degree_dist = degree_dist
        # This degree distribution is found in https://ieeexplore.ieee.org/document/7736044
        self.replicas = degree_distribution[self.degree_dist]["replicas"]
        self.replicas_prob = degree_distribution[self.degree_dist]["replicas_prob"]
        self.frame_length = num_slot_per_frame // num_subcarriers
        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma,
                                      frame_length=self.frame_length, replicas=self.replicas, device=device)
        self.num_nodes = num_nodes
        self.num_slot_per_frame = num_slot_per_frame
        self.target_load = target_load
        self.irsa_model_name = irsa_model
        self.num_subcarriers = num_subcarriers
        self.quantile_to_track = quantile_to_track

        self.threshold_index = 0

        if irsa_model == "optimal":
            self.irsa_model = irsa_fast
        else:
            self.irsa_model = imperfect_irsa

    def reset(self):
        self.policy.reset()

    def select_thresholds(self, error_distribution):
        # Select multiple thresholds such that the expected load is equal to the target load
        thresholds = np.zeros(len(self.replicas), dtype=int)
        # Start from the highest threshold and find the expected load for each threshold
        for i in range(len(self.replicas)):
            expected_silent_nodes = error_distribution.sum(dim=0) # num_bins
            expected_load = (self.num_nodes - expected_silent_nodes) / self.num_slot_per_frame 
            # Find the threshold index such that the closest expected load to the target load is achieved
            threshold_index = torch.argmin(torch.abs(expected_load - (self.target_load*np.sum(self.replicas_prob[i:])))).item()
            thresholds[i] = threshold_index
        self.threshold_index = thresholds
        return thresholds
    
    def select_replicas(self, threshold_index):
        replicas = self.policy.get_action(thresholds=threshold_index)
        return replicas
    
    def select_slots(self, replicas):
        # Select the slots for the nodes based on their replicas
        replicas = replicas.cpu().numpy()
        nodes_to_transmit = (replicas > 0).nonzero()[0]
        first_replica_slot = torch.zeros(self.num_nodes, dtype=torch.long, device=self.policy.process.device)
        slot_selection = np.zeros((self.num_nodes, self.num_slot_per_frame))

        
        for i in nodes_to_transmit:
            if replicas[i] > 0:
                if self.num_subcarriers > 1:
                    selected_slots = np.random.choice(self.frame_length, size=replicas[i], replace=False)
                    selected_carriers = np.random.choice(self.num_subcarriers, size=replicas[i], replace=True)
                    slot_selection[i, selected_slots + selected_carriers*self.frame_length] = 1
                    first_replica_slot[i] = selected_slots[0]
                else:
                    selected_slots = np.random.choice(self.num_slot_per_frame, size=replicas[i], replace=False)
                    slot_selection[i, selected_slots] = 1
                    first_replica_slot[i] = selected_slots[0]
        return slot_selection, first_replica_slot

    def simulate_frame(self, slot_selection):
        return self.irsa_model(slot_selection)
    
    def step(self):
        error_distribution = self.policy.get_error_distribution()
        threshold_index = self.select_thresholds(error_distribution)
        replicas = self.select_replicas(threshold_index)
        slot_selection, first_replica_slot = self.select_slots(replicas)
        observation = self.simulate_frame(slot_selection)

        # Some IRSA metrics
        self.tx_attempts = np.sum(replicas.cpu().numpy() > 0)
        # print((replicas == 8).sum().item())
        self.successful_decodings = np.sum(observation == 2)
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        
        index_of_decoded_nodes = (observation == 2).nonzero()[0]

        self.policy.update(observation=torch.from_numpy(observation).to(self.policy.process.device), 
                           threshold_index=threshold_index[0],
                           index_of_decoded_nodes=index_of_decoded_nodes,
                           first_replica_slot=first_replica_slot[index_of_decoded_nodes])
        
        # compute the error prob per replica
        probs_per_replica = np.zeros(len(self.replicas))
        nodes_which_transmitted = (replicas > 0).nonzero().cpu().numpy()
        not_received_nodes = (observation[nodes_which_transmitted] == 0).nonzero()[0]
        for i in range(len(self.replicas)):
            nodes_with_replica_i = (replicas.cpu().numpy()[nodes_which_transmitted] == self.replicas[i]).nonzero()[0]
            if len(nodes_with_replica_i) > 0:
                collision_prob_per_replica_i = len(np.intersect1d(nodes_with_replica_i, not_received_nodes)) / len(nodes_with_replica_i)
                # print(f"Collision prob for replica {self.replicas[i]}: {collision_prob_per_replica_i}")
                probs_per_replica[i] = collision_prob_per_replica_i

        # print(torch.max(self.policy.process.get_error()).item())
        

        return index_of_decoded_nodes, first_replica_slot[index_of_decoded_nodes], probs_per_replica

    def run(self, num_episodes: int, num_steps: int):
        rewards = np.zeros(num_episodes*num_steps)
        quantile_rewards = np.zeros(num_episodes*num_steps)

        metrics = {
            "throughput": np.zeros(num_episodes*num_steps),
            "collision": np.zeros(num_episodes*num_steps),
            "load": np.zeros(num_episodes*num_steps)
        }
        trace = np.zeros((num_episodes, num_steps, len(self.replicas) + 1)) # (num_episodes, num_steps, [th x2, collision])
        collision_prob_per_replica = np.zeros(len(self.replicas))


        for episode in range(num_episodes):
            self.reset()
            episode_reward = 0
            episode_quantile_reward = 0
            for step in range(num_steps):
                
                
                index_of_decoded_nodes, first_replica_slot, collision_prob_per_replica_t = self.step()     
                # # All the other nodes
                # not_selected_nodes = torch.ones(self.policy.process.num_nodes, dtype=torch.bool, device=self.policy.process.device)
                # not_selected_nodes[index_of_decoded_nodes] = False
                # r = -(self.policy.process.get_error()[not_selected_nodes]**2).sum()
                # if len(index_of_decoded_nodes) > 0:
                #     r -= (self.policy.process.get_error()[index_of_decoded_nodes]**2).sum() * ((((self.frame_length + 3 - first_replica_slot).mean().item())+1)/(2*(self.frame_length+3)))
                # episode_reward += (r/self.num_nodes).item()
                error = self.policy.process.get_error()**2
                reward = -torch.mean(error).item()                
                episode_reward += reward
                quantile_reward = -torch.quantile(error, self.quantile_to_track).item()
                episode_quantile_reward += quantile_reward

                metrics["throughput"][episode * num_steps + step] = self.throughput
                metrics["collision"][episode * num_steps + step] = self.collision
                metrics["load"][episode * num_steps + step] = self.load

                trace[episode, step, :-1] = self.threshold_index
                trace[episode, step, -1] = self.collision

                collision_prob_per_replica += collision_prob_per_replica_t

            rewards[episode * num_steps:(episode + 1) * num_steps] = episode_reward
            quantile_rewards[episode * num_steps:(episode + 1) * num_steps] = episode_quantile_reward
            # print(max(self.policy.process.get_error().cpu().numpy()), metrics["load"][-1])
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_quantile_reward": np.mean(quantile_rewards),
            "std_quantile_reward": np.std(quantile_rewards),
            "metrics": metrics,
            "cppr": collision_prob_per_replica / (num_steps * num_episodes),
            "trace": trace
        }
    


class AdaptiveIRSA:
    def __init__(self, 
                 num_nodes: int, 
                 dim: int, 
                 n_bins: int, 
                 max_value: float, 
                 sigma: float, 
                 num_slot_per_frame: int,
                 target_load: float,
                 device: torch.device,
                 irsa_model: str = "optimal",
                 degree_dist: str = "bcsa",
                 num_subcarriers: int = 1,
                 quantile_to_track: float = 0.95):
        
        self.frame_length = num_slot_per_frame // num_subcarriers

        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma, replicas=[1], 
                                      frame_length=self.frame_length, device=device)
        self.quantile_to_track = quantile_to_track
        self.num_nodes = num_nodes
        self.num_slot_per_frame = num_slot_per_frame
        self.target_load = target_load
        self.irsa_model_name = irsa_model
        self.num_subcarriers = num_subcarriers
        if irsa_model == "optimal":
            self.irsa_model = irsa_fast
        else:
            self.irsa_model = imperfect_irsa
        self.degree_dist = degree_dist

        self.collision = 0.0
        

        # This degree distribution is found in https://ieeexplore.ieee.org/document/7736044
        self.replicas = degree_distribution[self.degree_dist]["replicas"]
        self.replicas_prob = degree_distribution[self.degree_dist]["replicas_prob"]

        self.threshold_index = 0
        self.threshold = 0

    def reset(self):
        self.policy.reset()

    def select_threshold(self, error_distribution):
        if self.collision > 0.0: coll = 1
        else: coll = 0
        self.threshold = self.threshold + 0.03 * (coll - 1e-2)
        return int(self.threshold * self.policy.process.n_bins//2)
    
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
        first_replica_slot = torch.zeros(self.num_nodes, dtype=torch.long, device=self.policy.process.device)
        slot_selection = np.zeros((self.num_nodes, self.num_slot_per_frame))
        for i in range(self.num_nodes):
            if replicas[i] > 0:
                if self.num_subcarriers > 1:
                    selected_slots = np.random.choice(self.frame_length, size=replicas[i], replace=False)
                    selected_carriers = np.random.choice(self.num_subcarriers, size=replicas[i], replace=True)
                    slot_selection[i, selected_slots + selected_carriers*self.frame_length] = 1
                    first_replica_slot[i] = selected_slots[0]
                else:
                    selected_slots = np.random.choice(self.num_slot_per_frame, size=replicas[i], replace=False)
                    slot_selection[i, selected_slots] = 1
                    first_replica_slot[i] = selected_slots[0]
        return slot_selection, first_replica_slot

    def simulate_frame(self, slot_selection):
        return self.irsa_model(slot_selection)
    
    def step(self):
        error_distribution = self.policy.get_error_distribution()
        threshold_index = self.select_threshold(error_distribution)
        replicas = self.select_replicas(threshold_index)
        slot_selection, first_replica_slot = self.select_slots(replicas)
        observation = self.simulate_frame(slot_selection)
        # Some IRSA metrics
        self.tx_attempts = np.sum(replicas.cpu().numpy() > 0)
        self.successful_decodings = np.sum(observation == 2)
        index_of_decoded_nodes = (observation == 2).nonzero()[0]
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        self.policy.update(observation=torch.from_numpy(observation).to(self.policy.process.device), 
                           threshold_index=threshold_index, 
                           index_of_decoded_nodes=index_of_decoded_nodes,
                           first_replica_slot=first_replica_slot[index_of_decoded_nodes])
        return index_of_decoded_nodes, first_replica_slot[index_of_decoded_nodes]
    
    def run(self, num_episodes: int, num_steps: int):
        rewards = []
        quantile_rewards = []

        metrics = {
            "throughput": [],
            "collision": [],
            "load": []
        }

        for episode in range(num_episodes):
            self.reset()
            episode_reward = 0
            episode_quantile_reward = 0
            for step in range(num_steps):            
                index_of_decoded_nodes, first_replica_slot = self.step()     
                # # All the other nodes
                # not_selected_nodes = torch.ones(self.policy.process.num_nodes, dtype=torch.bool, device=self.policy.process.device)
                # not_selected_nodes[index_of_decoded_nodes] = False
                # r = -(self.policy.process.get_error()[not_selected_nodes]**2).sum()
                # if len(index_of_decoded_nodes) > 0:
                #     r -= (self.policy.process.get_error()[index_of_decoded_nodes]**2).sum() * (((((self.frame_length + 3) - first_replica_slot).mean().item())+1)/(2*(self.frame_length+3)))
                # episode_reward += (r/self.num_nodes).item()
                reward = -torch.mean(self.policy.process.get_error()**2).item()
                episode_reward += reward
                quantile_reward = -torch.quantile(self.policy.process.get_error()**2, self.quantile_to_track).item()
                episode_quantile_reward += quantile_reward
                metrics["throughput"].append(self.throughput)
                metrics["collision"].append(self.collision)
                metrics["load"].append(self.load)
            rewards.append(episode_reward)
            quantile_rewards.append(episode_quantile_reward)
            # print(max(self.policy.process.get_error().cpu().numpy()), metrics["load"][-1])
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_quantile_reward": np.mean(quantile_rewards),
            "std_quantile_reward": np.std(quantile_rewards),
            "metrics": metrics
        }



class DummyIRSA:
    def __init__(self, 
                 num_nodes: int, 
                 dim: int, 
                 n_bins: int, 
                 max_value: float, 
                 sigma: float, 
                 num_slot_per_frame: int,
                 target_load: float,
                 device: torch.device,
                 irsa_model: str = "optimal",
                 degree_dist: str = "bcsa"):
        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma, replicas=[1], device=device)
        self.num_nodes = num_nodes
        self.num_slot_per_frame = num_slot_per_frame
        self.target_load = target_load
        self.irsa_model_name = irsa_model
        if irsa_model == "optimal":
            self.irsa_model = optimal_irsa
        else:
            self.irsa_model = imperfect_irsa
        self.degree_dist = degree_dist

        # This degree distribution is found in https://ieeexplore.ieee.org/document/7736044
        self.replicas = degree_distribution[self.degree_dist]["replicas"]
        self.replicas_prob = degree_distribution[self.degree_dist]["replicas_prob"]

    def reset(self):
        self.policy.reset()
    
    def select_replicas(self):
        # Sample nodes with prob target_load * num_slot_per_frame / num_nodes
        prob = self.target_load * self.num_slot_per_frame / self.num_nodes
        nodes_above_threshold = torch.bernoulli(torch.full((self.num_nodes,), prob, device=self.policy.process.device)).nonzero(as_tuple=True)[0]
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

    def simulate_frame(self, slot_selection):
        return self.irsa_model(slot_selection)
    
    def step(self):
        # error_distribution = self.policy.get_error_distribution()
        replicas = self.select_replicas()
        slot_selection = self.select_slots(replicas)
        observation = self.simulate_frame(slot_selection)

        # Some IRSA metrics
        self.tx_attempts = np.sum(replicas.cpu().numpy() > 0)
        self.successful_decodings = np.sum(observation == 2)
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        self.policy.update(observation=torch.from_numpy(observation).to(self.policy.process.device), threshold_index=-1)

    def run(self, num_episodes: int, num_steps: int):
        rewards = []

        metrics = {
            "throughput": [],
            "collision": [],
            "load": []
        }

        for episode in range(num_episodes):
            self.reset()
            episode_reward = 0
            for step in range(num_steps):
                self.step()
                reward = -torch.mean(self.policy.process.get_error()**2).item()
                episode_reward += reward
                metrics["throughput"].append(self.throughput)
                metrics["collision"].append(self.collision)
                metrics["load"].append(self.load)
            rewards.append(episode_reward)
        return np.mean(rewards), np.std(rewards), metrics
    


class TargetIRSAnoImplicit:
    def __init__(self, 
                 num_nodes: int, 
                 dim: int, 
                 n_bins: int, 
                 max_value: float, 
                 sigma: float, 
                 num_slot_per_frame: int,
                 target_load: float,
                 device: torch.device,
                 irsa_model: str = "optimal",
                 degree_dist: str = "bcsa"):
        self.policy = ThresholdPolicy(num_nodes, dim, n_bins, max_value, sigma, replicas=[1], device=device)
        self.num_nodes = num_nodes
        self.num_slot_per_frame = num_slot_per_frame
        self.target_load = target_load
        self.irsa_model_name = irsa_model
        if irsa_model == "optimal":
            self.irsa_model = irsa_fast
        else:
            self.irsa_model = imperfect_irsa
        self.degree_dist = degree_dist

        # This degree distribution is found in https://ieeexplore.ieee.org/document/7736044
        self.replicas = degree_distribution[self.degree_dist]["replicas"]
        self.replicas_prob = degree_distribution[self.degree_dist]["replicas_prob"]

    def reset(self):
        self.policy.reset()

    def select_threshold(self, error_distribution):
        # Select the threshold such that the expected load is equal to the target load
        expected_silent_nodes = error_distribution.sum(dim=0) # num_bins
        expected_load = (self.num_nodes - expected_silent_nodes) / self.num_slot_per_frame 
        # Find the threshold index such that the closest expected load to the target load is achieved
        threshold_index = torch.argmin(torch.abs(expected_load - self.target_load)).item()

        variance = (error_distribution[:,threshold_index] * (1 - error_distribution[:,threshold_index])).sum().item()

        real_load = (self.policy.process.get_error() > self.policy.process.centers[self.policy.process.n_bins//2 + threshold_index]).sum().item() / self.num_slot_per_frame
        bias = real_load - expected_load[threshold_index].item()

        return threshold_index 
    
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

    def simulate_frame(self, slot_selection):
        return self.irsa_model(slot_selection)
    
    def step(self):
        error_distribution = self.policy.get_error_distribution()
        threshold_index = self.select_threshold(error_distribution)
        replicas = self.select_replicas(threshold_index)
        slot_selection = self.select_slots(replicas)
        observation = self.simulate_frame(slot_selection)
        # Set the nodes not received to zero
        observation[observation == 1] = 0

        # Some IRSA metrics
        self.tx_attempts = np.sum(replicas.cpu().numpy() > 0)
        self.successful_decodings = np.sum(observation == 2)
        if self.tx_attempts > 0:
            self.collision = (self.tx_attempts - self.successful_decodings) / self.tx_attempts
        else:
            self.collision = 0.0
        if self.collision > 1.0:
            self.collision = 1.0
        self.throughput = self.successful_decodings / self.num_slot_per_frame
        self.load = self.tx_attempts / self.num_slot_per_frame

        self.policy.update(observation=torch.from_numpy(observation).to(self.policy.process.device), threshold_index=threshold_index)

    def run(self, num_episodes: int, num_steps: int):
        rewards = []

        metrics = {
            "throughput": [],
            "collision": [],
            "load": []
        }

        for episode in range(num_episodes):
            self.reset()
            episode_reward = 0
            for step in range(num_steps):
                
                self.step()
                
                reward = -torch.mean(self.policy.process.get_error()**2).item()
                episode_reward += reward
                metrics["throughput"].append(self.throughput)
                metrics["collision"].append(self.collision)
                metrics["load"].append(self.load)
            rewards.append(episode_reward)
        return np.mean(rewards), np.std(rewards), metrics