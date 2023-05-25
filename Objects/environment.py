import gym
import numpy as np

class Environment(gym.Env):
    def __init__(self,graph):
        super().__init__(graph)
        self.graph = graph
        self.observation_shape = np.concatenate(list(graph.get_states().values()))
