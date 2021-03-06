# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:26:43 2017

@author: bbrattol
"""
import numpy as np
import sys, os
import argparse

import gym
from gym import wrappers

import torch

from PolicyNet import Policy

sys.path.append('utils')

import multiprocessing
CORES = int(float(multiprocessing.cpu_count())*0.75)

env = gym.make('CartPole-v0')
outdir = '/tmp/MountainCarContinuous-agent-results'
#env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(1)
torch.manual_seed(1)

#####################################################

parser = argparse.ArgumentParser(description='Train policy network to solve a Gym problem.')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate policy net')
parser.add_argument('--past', default=1, type=int, help='number of past time steps to use in input')
parser.add_argument('--fold', default='debug', type=str, help='checkpoint folder')
parser.add_argument('--render', action='store_true',help='render the environment')
args = parser.parse_args()

GPU_ID = args.gpu
LR = args.lr
past = args.past
checkfold = args.fold

if GPU_ID is not None:
    USE_GPU = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_ID)

########### set some parameters ##################
checkpointFold = 'LunaLander'+checkfold+'/'
if not os.path.exists(checkpointFold+'/logs_policy'):
    os.makedirs(checkpointFold+'/logs_policy')

########## Initialize networks ##########
S = env.observation_space.shape[0]
try:
    A = env.action_space.shape[0]
except:
    A = env.action_space.n

policy = Policy(actions=A,encode_size=S*past,LR=LR,use_gpu=USE_GPU,drop_episode=20)
print 'initialized Policy network'

########## Train model #####################
running_reward = 10
T = 10000
for i_episode in range(10000):
    state = env.reset()
    batch = [np.zeros([1,state.shape[0]])]*(past-1)
    batch.append(state[np.newaxis,:])
    
    t=-1
    done=False
    while(not done):
        t+=1
        state = np.concatenate(batch,axis=0).reshape([-1])
        state = torch.from_numpy(state)
        action = policy.select_action(state,i_episode)        
        
        state, reward, done, _ = env.step(action)
        if i_episode%50==0 and i_episode>0 and args.render:
            env.render()
        
        policy.rewards.append(reward)
        
        batch.append(state[np.newaxis,:])
        if len(batch)>=past:
            del batch[0]
    
    running_reward = running_reward * 0.99 + t * 0.01
    d_reward, value = policy.finish_episode()
    
    if i_episode%50==0:
        print 'Episode %5d, %3d iter, Reward %3.1f, discounted Reward %3.1f'%(i_episode,t+1, running_reward, np.mean(d_reward))
    
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break