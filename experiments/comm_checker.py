import numpy as np


class CommChecker(object):
    def __init__(self, num_agents, comm_check_rate=10, comm_dim=3):
        self.num_agents = num_agents
        self.communications_matches = np.zeros(num_agents)
        self.communications_matches_count = 0
        self.communications_matches_matrix = np.zeros([num_agents, comm_dim, comm_dim])  # env.world.dim_c == 3
        self.comm_check_rate = comm_check_rate
        self.comm_dim = comm_dim

    def check(self, observations, agents, episode_step):
        if episode_step % self.comm_check_rate != 0:
            return
        # matches_results = [0. for _ in range(len(agents))]
        matches_results = np.zeros(self.num_agents)
        for i, obs, agent, matrix, matches_result \
                in zip(range(self.num_agents), observations, agents, self.communications_matches_matrix, matches_results):
            if agent.silent:
                continue
            obs = obs[:self.comm_dim]
            obs_max_index = obs.argmax()
            comm_action = agent.action.c
            max_comm_action = np.zeros(len(comm_action))
            max_comm_action[comm_action.argmax()] = 1.

            # if np.all(matrix[obs_max_index] * max_comm_action == 0):
            #     matrix[obs_max_index] = max_comm_action
            #     matches_results[i] = .0
            # else:
            #     matches_results[i] = 1

            # determinant
            matrix[obs_max_index] = max_comm_action
            matches_results[i] = np.linalg.det(matrix)

        self.communications_matches_count += 1
        self.communications_matches = self.communications_matches + matches_results

    def get_result(self):
        result = self.communications_matches / self.communications_matches_count

        self.communications_matches = np.zeros(self.num_agents)
        self.communications_matches_count = 0

        return result
