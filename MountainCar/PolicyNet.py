# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:22:00 2017

@author: ubuechle
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

class Policy():
    def __init__(self, actions=12,encode_size=1024,LR=0.01,use_gpu=False,drop_episode=5):
        self.use_gpu = use_gpu
        self.encode_size = encode_size
        self.drop_episode = drop_episode
        self.gamma = 0.99
        self.actions = actions
        
        #self.net = PolicyNetLstm(encode_size,actions,use_gpu)
        self.net = PolicyNet(encode_size,actions,use_gpu)
        if use_gpu: 
            self.net.cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        #self.optimizer = optim.SGD(self.net.parameters(),lr=LR,momentum=0.9,weight_decay = 0.0005)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        
        self.saved_state_values = []
        self.saved_actions = []
        self.rewards = []
        
        self.episode_time = 0.0
    
    def finish_episode(self):
        R = 0
        
        discounted_rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        value_loss = 0
        rewards = torch.Tensor(discounted_rewards)
        
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for action, reward, value in zip(self.saved_actions, rewards, self.saved_state_values):
            if np.isnan(reward): reward=0
            r = reward - value.data[0]
            r = int(r)#torch.Tensor(r).cuda()
            action.reinforce(r)
            value_loss += self.value_criterion(value, Variable(torch.Tensor([reward])).cuda())
        
        #torch.cuda.LongTensor of size 1x1 (GPU 0)
        #print value_loss
        self.optimizer.zero_grad()
        nodes = [value_loss.view(1,1).type(torch.cuda.LongTensor)] + self.saved_actions
        gradients = [torch.ones(1)] + [None] * len(self.saved_actions)
        #nodes = [value_loss] + self.saved_actions
        #print nodes.size()
        autograd.backward(nodes, gradients)
        self.optimizer.step()
        
        del self.rewards[:]
        del self.saved_actions[:]
        del self.saved_state_values[:]
        
        return discounted_rewards, value_loss.cpu().data.numpy()
    
    def select_action(self,state,step):
        drop = max(0.35-float(step)/(self.drop_episode*3),0.00)
        
        action_score, state_value = self.__forward(state)
        
        action_score = F.dropout(action_score, drop, True) # Dropout for exploration
        action_score = F.softmax(action_score)
        action = action_score.multinomial(self.actions,replacement=False)
        
        self.saved_actions.append(action)
        self.saved_state_values.append(state_value)
        
        return action.cpu().data.numpy()
    
    def save(self,checkpointFold,episode):
        state = {
            'episode': episode,
            'state_dict': self.net.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }
        filename = '%s/policy_episode_%03i.pth.tar'%(checkpointFold,episode)
        torch.save(state, filename)
    
    def load(self,checkpoint):
        state = torch.load(checkpoint)
        self.net.load_state_dict(state['state_dict'])
    
    def __forward(self,state):
        self.net.init_hidden()
        
        if type(state) is np.ndarray:
            state = torch.from_numpy(state)
        
        if not type(state) is torch.autograd.variable.Variable:
            state = Variable(state)
        
        state = state.float()
        if self.use_gpu:
            state = state.cuda()
        
        out = self.net(state)
        return out
    
    def __batch(self,data):
        return data


class PolicyNet(nn.Module):
    
    def __init__(self, encode_size, actions=12,use_gpu=None):
        super(PolicyNet, self).__init__()
        self.encode_size = encode_size
        self.use_gpu = use_gpu
        
        fc_in_hidden = 128
        fc_a_hidden  = 32
        fc_v_hidden  = 32
#        fc_a_hidden  = fc_in_hidden
#        fc_v_hidden  = fc_in_hidden
        
        self.fc_in = nn.Sequential()
        self.fc_in.add_module('fc1',nn.Linear(self.encode_size, fc_in_hidden))
        self.fc_in.add_module('relu1',nn.ReLU(inplace=True))
        
        self.fc_a = nn.Sequential()
        self.fc_a.add_module('fc_a',nn.Linear(fc_in_hidden,fc_a_hidden))
        self.fc_a.add_module('relu_a',nn.ReLU(inplace=True))
        
        self.action = nn.Sequential()
        self.action.add_module('action',nn.Linear(fc_a_hidden, actions))
        
        self.fc_v = nn.Sequential()
        self.fc_v.add_module('fc_v',nn.Linear(fc_in_hidden,fc_v_hidden))
        self.fc_v.add_module('relu_v',nn.ReLU(inplace=True))
#        
        self.value = nn.Sequential()
        self.value.add_module('value',nn.Linear(fc_v_hidden, 1))
    
    def forward(self, x):
        x = self.fc_in(x)
        
        ax = self.fc_a(x)
#        ax = x
        action_scores = self.action(ax)
        
        vx = self.fc_v(x)
#        vx = x
        state_value = self.value(vx)
        return action_scores, state_value
    
    def init_hidden(self):
        return

class PolicyNetLstm(nn.Module):
    
    def __init__(self, encode_size, actions=12,use_gpu=None):
        super(PolicyNetLstm, self).__init__()
        self.encode_size = encode_size
        self.use_gpu = use_gpu
        fc_in_hidden = 128
        
        self.hidden_dim = 128
        self.numLstm = 1
        self.init_hidden()
        
#        self.fc_in = nn.Sequential()
#        self.fc_in.add_module('fc_in',nn.Linear(self.encode_size, fc_in_hidden))
#        self.fc_in.add_module('relu_in',nn.ReLU(inplace=True))
#        self.fc_in.add_module('drop_in',nn.Dropout(p=0.2))
        
        self.lstm = nn.LSTM(self.encode_size,self.hidden_dim,self.numLstm,dropout=True)
        
        self.drop = nn.Dropout(p=0.25)
        #self.relu = nn.ReLU(inplace=True)
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier',nn.Linear(self.hidden_dim, actions+1))
    
    def forward(self, x):
        N = x.size()[0]
        #x = self.fc_in(x)
        x = x.view(N,1,-1) # (sequence=sequences in a batch, batch=1, features)
        xlstm,self.hidden = self.lstm(x,self.hidden)
        #xlstm = self.drop(xlstm)
        xlstm = F.relu(xlstm)
        x = self.classifier(xlstm[-1,:,:].view(1, -1))
        
        action_scores= x[:,:-1].view(-1)
        state_value  = x[:,-1].view(-1)
        return action_scores, state_value
    
    def init_hidden(self):
        if hasattr(self,'hidden'):
            del self.hidden
        v1 = Variable(torch.zeros(self.numLstm, 1, self.hidden_dim),requires_grad = False)
        v2 = Variable(torch.zeros(self.numLstm, 1, self.hidden_dim),requires_grad = False)
        if self.use_gpu:
            v1 = v1.cuda()
            v2 = v2.cuda()
        self.hidden = (v1,v2)
    
    