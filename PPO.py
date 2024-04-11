# -*- coding: utf-8 -*-
import os
import random
import numpy as np
# import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
# from collections import deque
import math
# from keras.layers import Input, Dense
# from keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import tensorflow
from drl_base_v7 import DRLBase
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras import activations
import argparse
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

def gini(x):

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
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), int(hidden_dim/2))
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=int(hidden_dim/2), nhead=4, batch_first= True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc2_ = torch.nn.Linear(int(hidden_dim/2), hidden_dim)
        self.fc3_ = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4_ = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.V = torch.nn.Parameter(torch.randn((hidden_dim, 1), requires_grad=True))
        self.W = torch.nn.Parameter(torch.zeros((1, 1, hidden_dim), requires_grad=True))
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # x = x.unsqueeze(0)
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.nn.functional.normalize(x,dim=2)
        x = self.transformer_encoder(x)
        x = self.tanh(self.fc2_(x))
        x = self.dropout(x)
        # x = torch.cat((x, self.W), 1)
        x = self.tanh(self.fc4(x))
        a = self.fc4_(x)
        # a = x.matmul(self.V)
        return F.softmax(a, dim=1).squeeze(2)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.fc4 = torch.nn.Linear(72, 1)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=int(hidden_dim), nhead=4, batch_first= True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.dropout = torch.nn.Dropout(0.22)


    def forward(self, x):
        # x = x.unsqueeze(0)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        x = self.dropout(x)
        # x = self.transformer_encoder(x)
        x =self.fc3(x).squeeze(-1)
        return self.fc4(x).squeeze(0) # n x 1



class PPO_model:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim=7, hidden_dim=512, action_dim=73, actor_lr= 3e-4, critic_lr= 1e-3,
                 lmbda= 0.97, eps= 0.2, gamma= 0.99, device= torch.device("cuda") if torch.cuda.is_available() else torch.device( "cpu")):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def action_argmax(self, s, mask, tourist):
        if len(tourist.route_s) == 0:
            return tourist.startSpot

        else:
            rpt = [(math.cos((S[3]/S[0])*(0.5*math.pi))*S[1])/((S[2]+S[4]) ) for S in s] 
            for i in range(72):
                rpt[i] = rpt[i]*mask[i]
            return np.argmax(np.array(rpt))



    def take_action(self, state, mask, tourist, env):
        if len(tourist.route_s) == 0:
            return tourist.startSpot
        
        else:
            if tourist.remain_time > 10:
                if np.random.rand() < 0.1:
                    r = np.cos((state[:,3]/state[:,0])*(0.5*math.pi))*state[:,1]
                    t = state[:,2] + state[:,4]
                    action = r/t
                    return np.argmax(action * mask[:-1])

                else:
                    _mask = torch.tensor(mask).to(self.device).reshape(1,-1)[:,:-1]
                    # state = torch.tensor([state], dtype=torch.float).to(self.device).reshape(1,-1)
                    state = torch.tensor([state], dtype=torch.float).to(self.device)
                    probs = self.actor(state)
                    action_dist = torch.distributions.Categorical(probs * _mask)
                    action = action_dist.sample()
                    return action.item()
            else:
                _mask = torch.tensor(mask).to(self.device).reshape(1,-1)[:,:-1]
                # state = torch.tensor([state], dtype=torch.float).to(self.device).reshape(1,-1)
                state = torch.tensor([state], dtype=torch.float).to(self.device)
                probs = self.actor(state)
                action_dist = torch.distributions.Categorical(probs * _mask)
                action = action_dist.sample()
                return action.item()


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



    def update(self, transition_dict):
        states,actions,rewards,next_states,dones = self.remove_none(transition_dict) 
        if len(actions) != 0:
            states = torch.tensor(states,
                                dtype=torch.float).to(self.device)
            actions = torch.tensor(actions).view(-1, 1).to(
                self.device)
            rewards = torch.tensor(rewards,
                                dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(next_states,
                                    dtype=torch.float).to(self.device)
            dones = torch.tensor(dones,
                                dtype=torch.float).view(-1, 1).to(self.device)
            
            
            td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                        dones)
            td_delta = td_target - self.critic(states)
            advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                                td_delta.cpu()).to(self.device)
            old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()

            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

##########################################

class PPO(DRLBase):
    def __init__(self, env=None):
        super(PPO, self).__init__(env)

        self.model = PPO_model()

    def train(self, num_episodes):
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    transition_list= [{
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'rewards': [],
                        'dones': []
                    } for i_ in range(len(self.env.tourists))]

                    s_1 = self.env.reset(Yahoo_data_path = Yahoo_data_path) # 72 x 4
                    
                    terminal = False
                    while not terminal:
                        a_list = []

                        for i_t in range(len(self.env.tourists)): 
                            _s_1 = np.copy(s_1)
                            s_2 = np.ones((72,3))

                            for j in range(72):
                                s_2[j][0] = 4*(self.env.calculateDistance(i_t,self.env.spots[j])/self.env.tourists[i_t].speed) 
                                s_2[j][1] = self.env.tourists[i_t].remain_time 
                                s_2[j][2] = 32 
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
                                a  = self.model.take_action(s, mask, self.env.tourists[i_t],self.env)
                                # a  = self.model.action_argmax(s, mask, self.env.tourists[i_t])

                                while (a == self.env.tourists[i_t].endSpot and self.env.tourists[i_t].remain_time > 10) or a in self.env.tourists[i_t].route_s: 
                                    a  = self.model.take_action(s, mask, self.env.tourists[i_t],self.env)
                                    
                            else:
                                a = None
                            a_list.append(a)

                            transition_list[i_t]['states'].append(s) 
                            transition_list[i_t]['actions'].append(a) #

                            if a == None:
                                spot_ratio = None
                            else:
                                spot_ratio = self.env.spots[a].num/self.env.spots[a].cMax if a != len(self.env.spots) else None
                            
                            self.env.tourists[i_t].info_before.append(spot_ratio)
                        

                        next_s_1, rpd, r, done, congestion_reward_list = self.env.step(a_list)

                        for i_t in range(len(self.env.tourists)):
                            r_t = r[i_t]
                            congestion_reward = congestion_reward_list[i_t]

                            self.env.tourists[i_t].reward.append(r_t if r_t != None else 0)
                            self.env.tourists[i_t].congestion_reward.append(congestion_reward if congestion_reward != None else 0)

                        for i_t in range(len(self.env.tourists)):
                            _next_s_1 = np.copy(next_s_1)
                            next_s_2 = np.ones((72,3))

                            for j in range(72):
                                # next_s_2[j][3] = 1 if j in self.env.tourists[i_t].route_s else -1
                                next_s_2[j][0] = 4*(self.env.calculateDistance(i_t,self.env.spots[j])/self.env.tourists[i_t].speed) 
                                next_s_2[j][1] = self.env.tourists[i_t].remain_time 
                                next_s_2[j][2] = 32 
                                # next_s_2[j][4] = 1 if j == self.env.tourists[i_t].startSpot else 0
                                # next_s_2[j][5] = 1 if j == self.env.tourists[i_t].endSpot else 0
                            
                            _route_s = self.env.tourists[i_t].route_s
                            for _s in _route_s:
                                _next_s_1[_s][1] = 0.1 
                            
                            # next_s = np.concatenate((_next_s_1,next_s_2),axis = 1)
                            next_s = np.concatenate((next_s_1,next_s_2),axis = 1)

                            transition_list[i_t]['next_states'].append(next_s)
                            transition_list[i_t]['rewards'].append(rpd[i_t])
                            transition_list[i_t]['dones'].append(done[i_t])
                        s_1 = next_s_1
                        terminal = all(done)
                    
                    reward_all = 0
                    for t in self.env.tourists:
                        reward_all += sum(t.reward)

                    print('the total reweard is:', reward_all)

                    # 地图*起***********************************************************
                    iii = 0
                    if iii == 1:
                        ed=0
                        spots = self.env.spots
                        tourists = self.env.tourists
                        fig = plt.figure(figsize=(10, 10))
                        x=[s.location[0] for s in spots]
                        y=[s.location[1] for s in spots]
                        colors=np.random.uniform(15, 80, len(spots))
                        plt.scatter(x, y, marker='o', c=colors)

                        plt.text(0.03, 0.94 , "$ED_{r}$: "+str(round(ed, 2)), fontsize=30, transform = plt.gca().transAxes)                

                        for s in spots:
                            plt.annotate(s.spotId, (s.location[0], s.location[1]), fontsize=15)
                        plt.xticks(fontsize=25)
                        plt.yticks(fontsize=25)
                        plt.xticks([])
                        plt.yticks([])
                        for t in tourists:
                            tx=[spots[r].location[0] for r in t.route_s]
                            ty=[spots[r].location[1] for r in t.route_s]
                            x_length=[tx[i+1]-tx[i] for i in range(len(tx)-1)]
                            y_length=[ty[i+1]-ty[i] for i in range(len(ty)-1)]
                            try:
                                del tx[-1]
                                del ty[-1]
                                plt.quiver(tx, ty, x_length, y_length, color="C"+str(t.touristId), angles='xy', scale_units='xy', scale=1, width=0.005, alpha=0.6)
                            except:
                                print("PROBLEM OCCUR:")
                                print("TX::: ", tx)
                                print("TY::: ", ty)
                                print("x_length::: ", x_length)
                                print("y_length::: ", y_length)
                                print("T.ROUTE::: ", t.route)
                        dir="visual"
                        pp = PdfPages(dir+"/_map.pdf")
                        plt.margins(0.02,0.02)
                        pp.savefig(fig, bbox_inches = 'tight', pad_inches = 0)
                        pp.close()

                    # 地图*止***********************************************************


                    # 保存model
                    save_model = False
                    if save_model:
                        reward_record_list = [2550]
                        reward_record_list.append(int(reward_all))
                        max_reward = max(reward_record_list)
                        if int(reward_all) > max_reward:
                            model_name = str(int(reward_all))
                            dir = './' + model_name + '.pth'
                            torch.save(self.model.actor,dir)



                    # # # # 景点柱状图***********************************************************
                    bar = plt.figure(figsize=(20, 12.36))
                    x=[s.spotId for s in self.env.spots]
                    y=[0 for i in range(len(self.env.spots))]
                    c=[s.cMax for s in self.env.spots]

                    for t in self.env.tourists:
                        for r in t.route_s:
                            y[r]+=1.0

                    plt.bar(x, y, color='c')
                    # plt.legend(fontsize=50)
                    plt.xlabel("POI", fontsize=40)
                    plt.ylabel("Number of Visit", fontsize=40)
                    plt.xticks(fontsize=40)
                    yspot=[0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
                    plt.ylim([yspot[0],yspot[-1]])
                    plt.yticks(yspot, [str(y) for y in yspot], fontsize=40)
                    # plt.text(0.03, 0.82, info, fontsize=40, transform = plt.gca().transAxes)
                    
                    pp = PdfPages("./_spot_bar_500.pdf")
                    plt.margins(0.02,0.02)
                    pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
                    pp.close()
                    # # # # 景点柱状图***********************************************************

                    gini([s.visit_num/s.cMax for s in self.env.spots])
                    gini([sum(t.congestion_reward) for t in self.env.tourists])
                    sum([sum(t.congestion_reward) for t in self.env.tourists])
                    sum([sum(t.reward) for t in self.env.tourists])
                    
                    ########## 打印result ###########
                    print('spot gini ',gini([s.visit_num/s.cMax for s in self.env.spots]))
                    print('tourist gini',gini([sum(t.congestion_reward) for t in self.env.tourists]))
                    print('reward ',sum([sum(t.reward) for t in self.env.tourists]))
                    print('cong reward ',sum([sum(t.congestion_reward) for t in self.env.tourists]))
                    print('average_variance_spot_attend_ratio', np.average(self.env.spot_attend_ratio))
                    print('max_variance_spot_attend_ratio', max(self.env.spot_attend_ratio))
                    ########## 打印result ###########


                    # 更新model
                    for ii in range(len(self.env.tourists)):
                        self.model.update(transition_list[ii])
                    
                    if (i_episode + 1) % 100 == 0:
                        pbar.set_postfix({
                            'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return':
                            '%.3f' % 1
                        })
                    pbar.update(1)
##########################################

Yahoo_data_path = './data/aid_count.csv'
actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 30
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

ts = time.time()
now = datetime.now()
timestamp=now.strftime('%Y.%m.%d_%H_%M')
dir="visual"
model = PPO()

epoch=int(num_episodes)
# model.model.actor = torch.load('/home/y.kong/kyt_new/KYT_Route_recommendation/4615.pth')
# model.model.actor = torch.load('./pre_trained_model_6783.pth')
# model.model.actor = torch.load('/home/y.kong/kyt_new/KYT_Route_recommendation/4058_Yahoo.pth')
# model.model.actor = torch.load('/home/s2120431/Multiple_Tourist_Route_Planning_with_Dual_Congestion/self_att.pth')

model.train(epoch)
print('finish')
