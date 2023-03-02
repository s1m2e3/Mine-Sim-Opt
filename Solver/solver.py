
class Solver():
    def __init__(self,game_env):
        self.env=game_env
        self.inputs = self.env.observation_space
        self.actions = self.env.action_space
        self.brain = None

    def predict(self,state):
        action = self.brain(state)
        return action

class DQNAgent(Solver):
    def __init__(self, type):
        self.type = type


