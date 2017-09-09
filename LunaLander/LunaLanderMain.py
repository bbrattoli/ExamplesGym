# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:26:43 2017

@author: bbrattol
"""
import numpy as np
import sys, os
import argparse
from itertools import count

import gym
import tensorflow as tf
import torch

from PolicyNet import Policy

sys.path.append('../')
from logger import Logger

import multiprocessing
CORES = int(float(multiprocessing.cpu_count())*0.75)

env = gym.make('LunaLander-v1')
env.seed(1)
torch.manual_seed(1)

#####################################################

parser = argparse.ArgumentParser(description='Train policy network to solve a Gym problem.')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate policy net')
parser.add_argument('--past', default=0.01, type=int, help='number of past time steps to use in input')
parser.add_argument('--save_fold', default='debug', type=str, help='checkpoint folder')
args = parser.parse_args()

GPU_ID = args.gpu
LR = args.lr
past = args.past
checkfold = args.save_fold

if GPU_ID is not None:
    USE_GPU = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_ID)

########### set some parameters ##################
checkpointFold = checkfold+'/'
if not os.path.exists(checkpointFold+'/logs_policy'):
    os.makedirs(checkpointFold+'/logs_policy')

logger_p = Logger(checkpointFold+'/logs_policy')

########## Initialize networks ##########
policy = Policy(actions=2,encode_size=4*past,LR=LR,use_gpu=USE_GPU)
print 'initialized Policy network'

########## Train model #####################
T = 10000
for i_episode in range(10):
    state = env.reset()
    batch = [np.zeros([1,state.shape[0]])]*(past-1)
    batch.append(state[np.newaxis,:])
    
    for t in range(T):
        state = np.concatenate(batch,axis=0).reshape([-1])
        action = policy.select_action(state,i_episode)        
        state, reward, done, _ = env.step(action)
        policy.rewards.append(reward)
        
        batch.append(state[np.newaxis,:])
        if len(batch)>=past:
            del batch[0]
        
        if t%10==0:
            print reward
        
        if done:
            break
    
    d_reward, value = policy.finish_episode()
    
    logger_p.scalar_summary('reward', reward, i_episode)
    logger_p.scalar_summary('state_value_loss', value, i_episode)
    
    if i_episode%50==0:
        print 'Episode %d, Done in %d iterations, Reward %.2f'%(i_episode,t+1, reward)
    