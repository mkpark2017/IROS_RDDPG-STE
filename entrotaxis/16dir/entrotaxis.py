import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

from scipy.stats import norm
from scipy.io import savemat

from gym_ste_v2.envs.common.particle_filter import ParticleFilter
from utils import *
#import gym_ste.envs

DEBUG = True
env = gym.make('gym_ste_v2:StePfConvInfotaxisExtMatEnv-v0')

obs = env.reset()

parser = argparse.ArgumentParser(description='Infotaxis')
args = parser.parse_args()
args.env_sig = obs[7]
args.sensor_sig_m = obs[6]
args.pf_num = int(obs[5])
args.court_ly = obs[4]
args.court_lx = obs[3]
args.gas_t = obs[2]
args.gas_d = obs[1]
args.model_path = '.'
args.mode = 'test'

export_parameter_to_text(args, env)

args.np_random = env.np_random
particle_filter = ParticleFilter(args)

dist = obs[0]

total_rewards = 0
total_steps = 0
total_q_diff = 0
success_episodes = 0

total_episodes = 1000

steps_list = []
episode_reward_list = []
q_diff_list = []
num_dir = 16

sim_time_mean = []

for e in range(total_episodes):
    if e != 0: obs = env.reset()
    cumulated_reward = 0
    steps = 0
    done = False
    #env.render_background(mode='human') # Draw backgound plot
    pf_num = args.pf_num
    etc_num = 17

    sim_time = []
    while not done and steps < 300:
    #while 1:
        start = time.time()

        steps += 1
        agent_x = obs[11]
        agent_y = obs[12]

        pf_x = obs[etc_num:etc_num+pf_num]
        pf_y = obs[etc_num+pf_num:etc_num+pf_num*2]
        pf_q = obs[etc_num+pf_num*2:etc_num+pf_num*3]
        Wpnorms = obs[etc_num+pf_num*3:etc_num+pf_num*4]

        measure = obs[15]
        hSig = math.sqrt( pow(measure*args.sensor_sig_m,2) + pow(args.env_sig,2) )
        if hSig == 0: hSig = 1e-10 #avoid zero

        max_gauss = measure + 3*hSig
        min_gauss = measure - 3*hSig
        if min_gauss < 0: min_gauss = 0
        num_interval = 20
        gauss_x = []
        for z in range(num_interval):
            gauss_x.append(min_gauss + z*(max_gauss-min_gauss)/(num_interval-1) )

        wind_d = obs[8]*math.pi
        #print("infotaxis_wind_d: ", wind_d)
        wind_s = obs[9]

        new_util = np.ones(num_dir)*np.nan
        #obs_temp = obs
        entropy_a = []

        for a in range(num_dir):
#            start2 = time.time()
            angle = (-1 + 2/num_dir*a)*math.pi
            #act = (-1 + 2/num_dir*a)
            #print(act)
            #[new_agent_x, new_agent_y] = np.ones(2)*np.nan
            new_agent_x = agent_x + math.cos(angle)*dist
            new_agent_y = agent_y + math.sin(angle)*dist

#            print("this_time: ", time.time()-start2)

            #if (new_agent_x <= 60*0.01 or new_agent_x >= 60*0.99 or new_agent_y <= 60*0.01 or new_agent_y >= 60*0.99):
            #print("new_x", new_agent_x, "new_y", new_agent_y)
            if new_agent_x <= 0 or new_agent_x >= 60 or new_agent_y <= 0 or new_agent_y >= 60:
                entropy_a.append(np.nan)
                continue
            else:
                entropy = 0
                zu_sum = []
                p_c = np.array([]) #probability of concentration
                for z in range(num_interval):
                    gauss_temp = particle_filter._weight_calculate(gauss_x[z], new_agent_x, new_agent_y,
                                                                   pf_x, pf_y, pf_q, wind_d, wind_s)
                    #gauss_temp = gauss_temp*(max_gauss-min_gauss)/(num_interval-1)
                    #if i>0:
                    #gauss_old = gauss_temp
                    zu_tot = Wpnorms*gauss_temp
                    p_c = np.append(p_c, sum(zu_tot))
                
                p_c_sum = np.sum(p_c)
#                print("Sum of probability of conetration: ", p_c_sum)
                p_c = p_c/p_c_sum
                entropy = -sum(p_c*np.log2(p_c+(p_c==0)*1) )
                entropy_a.append(entropy)
#                    zWpnorm = zu_tot/zu_sum[z]
                    #print(sum(zWpnorm))
        #print(entropy_a)
        a = np.nanargmax(entropy_a)
        #print("entropy: ", len(entropy_a))
        act = (-1 + 2/num_dir*a)

        #new_agent_x = agent_x + math.cos(act*math.pi)*dist
        #new_agent_y = agent_y + math.sin(act*math.pi)*dist
        #print("agnet_x: ", agent_x, "	| agent_y: ", agent_y)
        #print("new_agnet_x: ", new_agent_x, "	| new_agent_y: ", new_agent_y)

        sim_time_temp = time.time() - start
#        print(sim_time_temp)
        sim_time.append(sim_time_temp)
#        act = 1
        obs, rew, done, info = env.step(act)     # take a random action

        #env.render(mode='human') # Draw UAV plot
        cumulated_reward += rew
        #time.sleep(0.1)
#        if DEBUG:
#            print(info)
#        env.close()
    sim_time_temp = np.mean(sim_time)
    sim_time_mean.append(sim_time_temp)

    q_diff = abs(info[0]-info[1])
    print("Episode: ", e,  " | Episode step: ", steps, " | Reward: ", round(cumulated_reward,1), " | Sim Time: ", round(sim_time_temp,4), " | q_diff: ", q_diff)
    #if DEBUG and done:
    #    time.sleep(3)

    total_rewards += cumulated_reward
    if cumulated_reward >= 100:
        total_q_diff += q_diff
        total_steps += steps
        success_episodes += 1

    steps_list.append(steps)
    episode_reward_list.append(cumulated_reward)
    q_diff_list.append(q_diff)

#    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()

print("Steps_list: ", steps_list)
print("Reward_list: ", episode_reward_list)

print("Mean reward: ", total_rewards/total_episodes)
print("Mean steps: ", total_steps/success_episodes)
print("Q_diff: ", total_q_diff/success_episodes)

results_dic = {"steps_list": steps_list, "sim_time": sim_time_mean, "episode_reward_list": episode_reward_list, "q_diff_list": q_diff_list}
#reward_list_dic = {"episode_reward_list": episode_reward_list, "label": "experiment_rewards"}

savemat("test_data.mat", results_dic)
#savemat("reward_list_env04_sen02_24directions.mat", reward_list_dic)

