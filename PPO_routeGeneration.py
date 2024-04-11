# -*- coding: utf-8 -*-
import os
import random
import numpy as np
# import tensorflow as tf
# from matplotlib.backends.backend_pdf import PdfPages
# from collections import deque
import math
from os.path import dirname, abspath
# from keras.layers import Input, Dense
# from keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import tensorflow
from drl_base_v7_routeGeneration_ import DRLBase
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras import activations
import argparse
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
# import rl_utils
# from tqdm import tqdm
# import matplotlib.pyplot as plt

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

    
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, int(hidden_dim/2))
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        x = F.relu(self.fc4(F.relu(self.fc3(x))))
        return F.softmax(self.fc5(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)

personalize_score = True
rel_personalize_score_path = '/data_/personalize_spots_prob_01.csv'
ID = '01'
project_dir = dirname(abspath(__file__))
personalize_score_path = project_dir + rel_personalize_score_path

class PPO_model:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim=455, hidden_dim=100, action_dim=92, actor_lr= 3e-4, critic_lr= 1e-3,
                 lmbda= 0.97, eps= 0.2, gamma= 0.99, device = torch.device("cpu")):
        self.actor_DualCongestion = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_NO_DualCongestion = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
        #                                         lr=actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
        #                                          lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device


    def action_DualCongestion(self, state, mask, tourist):
        if len(tourist.route_s) == 0:
            return tourist.startSpot

        else:
            if tourist.remain_time > 10:
                # 以10%的概率去性价比最高的spot
                if np.random.rand() < 0:
                    r = np.cos((state[:,3]/state[:,0])*(0.5*math.pi))*state[:,1]
                    t = state[:,2] + state[:,4]
                    action = r/t
                    return np.argmax(action * mask[:-1])

                else:
                    _mask = torch.tensor(mask).to(self.device).reshape(1,-1)
                    state = torch.tensor([state], dtype=torch.float).to(self.device).reshape(1,-1)
                    probs = self.actor_DualCongestion(state)
                    action_dist = torch.distributions.Categorical(probs * _mask)
                    action = action_dist.sample()
                    return action.item()
                    # return (probs * _mask).argmax().item()
            else:
                _mask = torch.tensor(mask).to(self.device).reshape(1,-1)
                state = torch.tensor([state], dtype=torch.float).to(self.device).reshape(1,-1)
                probs = self.actor_DualCongestion(state)
                action_dist = torch.distributions.Categorical(probs * _mask)
                action = action_dist.sample()
                return action.item()
                # return (probs * _mask).argmax().item()


    def remove_none(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']
        dones = transition_dict['dones']

        while None in actions:
            for i in range(len(actions)):
                if actions[i] == None:
                    del states[i]
                    del actions[i]
                    del rewards[i]
                    del next_states[i]
                    del dones[i]
                    break
        return states,actions,rewards,next_states,dones

# ##########################################

class PPO(DRLBase):
    def __init__(self, env=None):
        super(PPO, self).__init__(env)

        self.model = PPO_model()
                    
    def generateRoute_DualCongestion(self, startSpot=0, endSpot=59, startTime=9, budget=8, personalize_score = False, personalize_score_path = personalize_score_path):
        s_1 = self.env.reset(start_time = startTime, startSpot = startSpot, endSpot = endSpot, budget = budget ,sim_data_path = sim_data_path, personalize_score=personalize_score, personalize_score_path=personalize_score_path)
        terminal = False
        while not terminal:
            a_list = []

            # 拿到每一个tourist的特征，并且得到action
            for i_t in range(len(self.env.tourists)): 
                _s_1 = np.copy(s_1)
                s_2 = np.ones((91,1))

                for j in range(91):
                    # s_2[j][3] = 1 if j in self.env.tourists[i_t].route_s else -1 # 记录是否去过这个spot
                    # s_2[j][0] = 4*(self.env.calculateDistance(i_t,self.env.spots[j])/self.env.tourists[i_t].speed) # 第 i 个tourist到第 j 个spot 的时间
                    s_2[j][0] = 4*(self.env.calculateTime(i_t,j)) # 第 i 个tourist到第 j 个spot 的时间
                    # s_2[j][1] = self.env.tourists[i_t].remain_time # 所剩时间
                    # s_2[j][2] = 32 # 游览总时间
                    # s_2[j][4] = 1 if j == self.env.tourists[i_t].startSpot else 0
                    # s_2[j][5] = 1 if j == self.env.tourists[i_t].endSpot else 0

                # 如果去过这个spot，那么把该spot的特征全部置0
                _route_s = self.env.tourists[i_t].route_s
                for _s in _route_s:
                    _s_1[_s][1] = 0.1 # 将去过的spot的分数设为很小的值

                # s = np.concatenate((_s_1,s_2),axis = 1) 
                s = np.concatenate((s_1,s_2),axis = 1)                        

                if self.env.tourists[i_t].unfrozen_time_step == self.env.current_time and self.env.tourists[i_t].complete == 0:
                    # mask = self.env.tourists[i_t].mask
                    mask = self.env.process_mask(i_t)
                    # a  = self.model.take_action(s, mask, self.env.tourists[i_t],self.env)
                    a  = self.model.action_DualCongestion(s, mask, self.env.tourists[i_t])

                    # while (a == self.env.tourists[i_t].endSpot and self.env.tourists[i_t].remain_time > 10) or a in self.env.tourists[i_t].route_s: # 避免刚开始就抽到endspot,以及重复spot
                    #     a  = self.model.action_DualCongestion(s, mask, self.env.tourists[i_t])
                        
                else:
                    a = None
                a_list.append(a)

                # transition_list[i_t]['states'].append(s) # 存入state
                # transition_list[i_t]['actions'].append(a) # 存入action


                ########## 查看做这个action的时候对应spot的人数比
                if a == None:
                    spot_ratio = None
                else:
                    spot_ratio = self.env.spots[a].num/self.env.spots[a].cMax if a != len(self.env.spots) else None
                
                self.env.tourists[i_t].info_before.append(spot_ratio)
                ##########
            

            next_s_1, rpd, r, done, congestion_reward_list = self.env.step(a_list)

            for i_t in range(len(self.env.tourists)):
                r_t = r[i_t]
                congestion_reward = congestion_reward_list[i_t]

                self.env.tourists[i_t].reward.append(r_t if r_t != None else 0)
                self.env.tourists[i_t].congestion_reward.append(congestion_reward if congestion_reward != None else 0)

            for i_t in range(len(self.env.tourists)):
                _next_s_1 = np.copy(next_s_1) 
                next_s_2 = np.ones((91,1))

                for j in range(91):
                    # next_s_2[j][3] = 1 if j in self.env.tourists[i_t].route_s else -1
                    # next_s_2[j][0] = 4*(self.env.calculateDistance(i_t,self.env.spots[j])/self.env.tourists[i_t].speed) # 第 i 个tourist到 第j 个spot 的时间
                    next_s_2[j][0] = 4*(self.env.calculateTime(i_t,j))
                    # next_s_2[j][1] = self.env.tourists[i_t].remain_time # 所剩时间
                    # next_s_2[j][2] = 32 # 游览总时间
                    # next_s_2[j][4] = 1 if j == self.env.tourists[i_t].startSpot else 0
                    # next_s_2[j][5] = 1 if j == self.env.tourists[i_t].endSpot else 0
                
                _route_s = self.env.tourists[i_t].route_s
                for _s in _route_s:
                    _next_s_1[_s][1] = 0.1 # 将去过的spot的分数设为很小的值
                
                # next_s = np.concatenate((_next_s_1,next_s_2),axis = 1)
                next_s = np.concatenate((next_s_1,next_s_2),axis = 1)

                # transition_list[i_t]['next_states'].append(next_s)
                # transition_list[i_t]['rewards'].append(rpd[i_t])
                # transition_list[i_t]['dones'].append(done[i_t])
            s_1 = next_s_1
            terminal = all(done)

        routeGeneration = self.env.tourists[0].route_name
        StartTimetoVisit = self.env.tourists[0].StartTimetoVisit
        spot_ratio = self.env.tourists[0].info_before
        route_ID = self.env.tourists[0].route_s

        return routeGeneration, StartTimetoVisit, spot_ratio, route_ID 

def minutes2hours(minutes):
    hours = float(minutes / 30) / 2
    return hours

def routeGen_DualCongestion(startSpot=0, endSpot=59, startTime=570, budget=480, personalize_score = personalize_score, ID = ID):
    h_startTime = minutes2hours(startTime)
    h_budget = minutes2hours(budget)
    if personalize_score:
        rel_personalize_score_path = "/data_/personalize_spots_prob_" + ID + ".csv"
        personalize_score_path = project_dir + rel_personalize_score_path
    route, StartTimetoVisit, _, route_id = model.generateRoute_DualCongestion(startSpot=startSpot, endSpot=endSpot, startTime=h_startTime, budget=h_budget, personalize_score = personalize_score, personalize_score_path = personalize_score_path)
    userID = ID if len(ID)!=0 else None 
    print('\nDualCongestion')
    print('personalize_userID:\n', userID)
    print('Recommend route:\n', route)
    print('route_id:\n', route_id)
    print('StartTimetoVisit:\n', StartTimetoVisit)
    return userID, route, StartTimetoVisit, route_id

if __name__ == "__main__":

    if personalize_score:
        ID = rel_personalize_score_path.split('_')[-1].split('.')[0]

    

    sim_data_path = project_dir + '/data_/sim_data_ave.csv'
    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 100
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2
    device = torch.device("cpu")
    epoch=int(num_episodes)

    ts = time.time()
    now = datetime.now()
    timestamp=now.strftime('%Y.%m.%d_%H_%M')
    dir="visual"
    print(dir)
    model = PPO()

    model.model.actor_DualCongestion = torch.load(project_dir + '/91spots_DualCongestion.pth', map_location=torch.device('cpu'))#.to(device)
    
    routeGen_DualCongestion()

