from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np
class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.keys = ["collision", "throughput", "load"]
        self.ep_reward = deque(maxlen=10)
        self.load = deque(maxlen=100*10)
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            for key, value in info.items():
                if key in self.keys:
                    self.logger.record_mean(f"metrics/{key}", float(value))
                if key == "load":
                    self.load.append(float(value))
                if key == "episode":
                    self.ep_reward.append(info["episode"]["r"])
                
        return True
        
    

def evaluate_policy(model, env, num_envs, n_time_steps=1000):
    all_rewards = []
    load = []
    obs = env.reset()
    for step in range(n_time_steps // num_envs):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        for info in infos:
            for key in info.keys():
                if key == "episode":
                    all_rewards.append(info[key]["r"])
                if key == "load":
                    load.append(float(info[key]))
    return np.mean(all_rewards), np.std(all_rewards), np.mean(load) 