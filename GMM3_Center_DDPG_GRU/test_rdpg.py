'''
Recurrent Deterministic Policy Gradient (DDPG with GRU network)
Update with batch of episodes for each time, so requires each episode has the same length.
'''
import time
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from common.buffers import *
from common.utils import *
from common.rdpg import *
from common.evaluator import *

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import argparse
from gym import spaces

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("Learning device: ", device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RDPG')
    # Set Environment
    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_ste_v2:StePfCentGmmConvExtMatEnv-v0', type=str, help='open-ai gym environment')

    # Set network parameter
    parser.add_argument('--hidden_1', default=256, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden_2', default=64, type=int, help='input num of GRU layer')
    parser.add_argument('--hidden_3', default=32, type=int, help='output num of GRU layer')
    parser.add_argument('--n_layers', default=1, type=int, help='number of stack for hidden layer')
    parser.add_argument('--rate', default=0.0002, type=float, help='Q learning rate')
    parser.add_argument('--prate', default=0.00002, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.0001, type=float, help='moving average for target network')
    parser.add_argument('--drop_prob', default=0.2, type=float, help='dropout_probability')

    # Set learning parameter
    parser.add_argument('--rbsize', default=100000, type=int, help='Memory size')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--blength', default=1, type=int, help='minibatch sequence length')
    parser.add_argument('--warmup', default=100000, type=int, help='warmup size (steps or episodes)')
    parser.add_argument('--max_episodes', default=200000, type=int, help='Number of episodes')
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode')
    parser.add_argument('--validate_episodes', default=20, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_interval', default=1000, type=int, help='Validation episode interval')
    parser.add_argument('--epsilon_rate', default=0.1, type=int, help='linear decay of exploration policy')

    #etc
    parser.add_argument('--pause_time', default=0, type=float, help='Pause time for evaluation')
    parser.add_argument('--model_path', default='model/2022_02_09_low_noise_current/gym_ste_v2:StePfCentGmmConvExtEnv-v0-run1/', type=str, help='Output root')
    parser.add_argument('--model_path_current', default='.', type=str, help='Output root')

    #CUDA GPU
    parser.add_argument('--device_idx', default=device_idx, help='cuda device num: -1(CPU), 0<= (GPU) ')


    args = parser.parse_args()

    env = gym.make(args.env)
    action_space = env.action_space
    state_space  = env.observation_space

    replay_buffer_size = args.rbsize
    replay_buffer = ReplayBufferGRU(replay_buffer_size)

    n_layers = args.n_layers
    torch.autograd.set_detect_anomaly(True)
    alg = RDPG(args, replay_buffer, state_space, action_space)


    if args.mode == 'test':
        export_parameter_to_text(args, env)
        test_episodes = 1000
        max_steps=300

        alg.load_model(args.model_path)
        total_reward = 0
        total_steps = 0
        total_q_diff = 0
        success_episodes = 0

        steps_list = []
        q_diff_list = []
        episode_reward_list = []
        sim_time_mean = []

        for episode in range(test_episodes):
            env.close()
            observation = env.reset()
            #env.render_background(mode='human') #Draw background plot
            episode_steps = 0
            episode_reward = 0.
            sim_time = []

            done = False

            hidden_out = torch.zeros([n_layers, 1, args.hidden_3], dtype=torch.float).cuda(device)
            steps = 0
            while not done:
                start = time.time()
                steps += 1
                hidden_in = hidden_out
                action, hidden_out= alg.policy_net.get_action(observation, hidden_in, noise_scale=0.0)  # no noise for testing
#                print("current action:" + str(action))
                observation, reward, done, info = env.step(action)
                if steps >= max_steps: done = True
                #env.render(mode='human') #Draw simulation plot
                sim_time_temp = time.time()-start
                sim_time.append(sim_time_temp)
                # update
                episode_reward += reward
                episode_steps += 1
                time.sleep(args.pause_time)
            sim_time_temp = np.mean(sim_time)
            sim_time_mean.append(sim_time_temp)

            q_diff = abs(info[0]-info[1])
            prRed('[Evaluate] Episode_{:07d}: | mean_reward:{} | sim_time:{} | q_diff:{}'.format(episode, round(episode_reward,1), round(sim_time_temp,4), q_diff) )
            print("Episode steps: ", steps)
            total_reward += episode_reward
            if episode_reward >= 100:
                total_steps += steps
                total_q_diff += q_diff
                success_episodes += 1

            steps_list.append(steps)
            q_diff_list.append(q_diff)
            episode_reward_list.append(episode_reward)


        mean_reward = total_reward/test_episodes
        prGreen('[Evaluate] Episode_{:07d}: mean_reward:{}'.format(episode, mean_reward))
        mean_steps = total_steps/success_episodes
        mean_q_diff = total_q_diff/success_episodes
        print("Mean steps: ", mean_steps)
        print("Q_diff: ", mean_q_diff)

        results_list_dic = {"steps_list": steps_list, "episode_reward_list": episode_reward_list, "sim_time": sim_time_mean, "q_diff": q_diff_list}

        savemat('{}/test_data'.format(args.model_path)+".mat", results_list_dic)


