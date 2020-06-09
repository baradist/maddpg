import pickle
import time
from pathlib import Path

import multiagent.scenarios as scenarios
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import visdom

import maddpg.common.tf_util as U
from experiments.parse_args import parse_args
from maddpg.trainer.maddpg import MADDPGAgentTrainer


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, benchmark=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = scenario.get_env(world, scenario.reset_world, scenario.reward, scenario.observation,
                           done_callback=scenario.done)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "adversary_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def load_state(load_dir):
    try:
        U.load_state(load_dir)
        print('Previous state loaded')
    except:
        print("can not load")


def process(arglist):
    if arglist.exp_name == None:
        arglist.exp_name = arglist.scenario
    save_dir = arglist.save_dir + arglist.exp_name + '/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = save_dir

    if arglist.display:
        play(arglist)
    else:
        train(arglist)


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = get_num_adversaries(env)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        from experiments.plotter import Plotter
        vis = visdom.Visdom(port=8097)
        title = arglist.scenario + " " + arglist.exp_name
        episode = 'Episode'
        reward_plotter = Plotter(vis,
                                 title=title,
                                 ylabel='Reward',
                                 xlabel=episode,
                                 legend=['Agent-%d' % i for i in range(env.n)])
        comm_plotter = Plotter(vis,
                               title=title,
                               ylabel='Consistency',
                               xlabel=episode,
                               legend=['Agent-%d' % i for i in range(env.n)],
                               frequency=10)
        time_plotter = Plotter(vis,
                               title=title,
                               ylabel='Time, sec',
                               xlabel=episode,
                               legend=['Time'], frequency=10)
        overall_time_plotter = Plotter(vis,
                                       title=title,
                                       ylabel='Time from the beginning',
                                       xlabel=episode,
                                       legend=['Time'], frequency=1)

        U.initialize()

        load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        episodes_count = 0
        t_start = time.time()
        t_start_p = time.time()
        t_start_overall = time.time()

        from experiments.comm_checker import CommChecker
        comm_checker = CommChecker(env.n, comm_dim=env.world.dim_c )

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            comm_checker.check(obs_n, env.world.agents, episode_step)
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                episodes_count += 1
                reward_plotter.log(episodes_count, rew_n)
                if episodes_count % 10 == 0:
                    time_plotter.log(episodes_count, [time.time() - t_start_p])
                    t_start_p = time.time()
                if episodes_count % 100 == 0:
                    overall_time_plotter.log(episodes_count, [time.time() - t_start_overall])
                comm_plotter.log(episodes_count, comm_checker.get_result())

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # update all trainers, if not in display mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (episodes_count % arglist.save_rate == 0):
                U.save_state(arglist.load_dir, saver=saver)  # TODO
                mean_episode_reward = np.mean(episode_rewards)
                episode_rewards = [0.0]
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, episodes_count, mean_episode_reward, round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, episodes_count, mean_episode_reward,
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(mean_episode_reward)
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if episodes_count > arglist.num_episodes:
                Path(arglist.plots_dir).mkdir(parents=True, exist_ok=True)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


def play(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = get_num_adversaries(env)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_info = [[[]]]  # placeholder for benchmarking info
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # # collect experience
            # for i, agent in enumerate(trainers):
            #     agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                print("train step: {}, episode reward: {}, time: {}".format(
                    train_step, np.mean(episode_rewards[-1:]), round(time.time() - t_start, 3)))
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            time.sleep(0.1)
            env.render()


def get_num_adversaries(env):
    return np.sum([a.adversary if hasattr(a, 'adversary') else False for a in env.agents])


if __name__ == '__main__':
    arglist = parse_args()
    process(arglist)
