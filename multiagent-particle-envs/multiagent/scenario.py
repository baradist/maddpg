import numpy as np

# defines scenario upon which the world is built
from multiagent.environment import MultiAgentEnv


class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world

    def reset_world(self, world):
        raise NotImplementedError()

    def get_env(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, discrete_action_space=True):
        return MultiAgentEnv(world, reset_callback, reward_callback,
                 observation_callback, info_callback,
                 done_callback, shared_viewer, discrete_action_space=discrete_action_space)

    def done(self, agent, world):
        return False
