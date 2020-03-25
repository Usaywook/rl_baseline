import gym

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    
#     def __init__(self, env):
#         super(NormalizedEnv, self).__init__(env)
        
    def _action(self, action):
        """restore action"""
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        """normalize action"""
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    
    def step(self, action):
        next_state, reward, done, info = self.unwrapped.step(action)
        return next_state, reward, done, info
    