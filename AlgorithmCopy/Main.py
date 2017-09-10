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

env = gym.make('Copy-v0')
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
S = 1
A = 20

policy = Policy(actions=A,encode_size=S*past,LR=LR,
                use_gpu=USE_GPU,drop_episode=1000)
print 'initialized Policy network'

########## Train model #####################
rewards = []
for i_episode in xrange(100000):
    state = env.reset()
    
    cumulative_reward = 0
    t=-1
    done=False
    while(not done):
        t+=1
        action = policy.select_action(torch.Tensor([state]),i_episode)
        action = np.unravel_index(action,[2,2,5])
        #action = (1,1,state)
        
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        
        policy.rewards.append(reward)
    
    rewards.append(cumulative_reward)
    if args.render and i_episode%50==0:
        env.render()
    
    d_reward, value = policy.finish_episode()
    
    if i_episode%50==0:
        print 'Episode %5d, %3d iter, Reward %3.1f, Discounted %3.1f'%(i_episode,t+1, np.mean(rewards[-100:]), np.mean(d_reward))
    
    if np.mean(rewards[-100:])>=25 and i_episode>100:
        print 'Solved in %d episodes'%i_episode
        break
