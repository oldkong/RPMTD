# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from rrEnv_v7_routeGeneration import RREnv
# from catrr import RREnv

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

class DRLBase:
    def __init__(self, env=None):
        if env is not None:
            self.env = env
        else:
            self.env = RREnv()
        # self.env = RREnv()
        # self.env = env

        if not os.path.exists('model'):
            os.mkdir('model')

        if not os.path.exists('history'):
            os.mkdir('history')

    # def egreedy(observation):
    #     if np.random.rand() <= 0.1: # 10%的概率，按照visit的次数，去没有被游玩过的景点。怎么去了？
    #         return pass
    #     else:  action = np.argmax(observation) # 90%的概率，去性价比最高的景点

    #     return action
        


    def play(self, m='pg'):
        """play game with model.
        """
        # print('play...')
        observation = self.env.reset()

        reward_sum = 0

        allComplete=1
        for tourist in self.env.tourists:
            allComplete *= tourist.complete        
        while allComplete == 0 and touristId !=-1:
            self.env.render()
            for tourist in self.env.tourists:
                allComplete *= tourist.complete
            action = np.argmax(observation) 
            # action = egreedy(observation)
            observation, reward, done, touristId = self.env.step([touristId, action])

            reward_sum += reward

        # print("-"*100)
        # for spot in self.env.spots:
        #     print(spot)
        # print("-"*100)
        # for tourist in self.env.tourists:
        #     print(tourist)
        # print("-"*100)
        reward=[]
        static_reward=[]
        for tourist in self.env.tourists:
            reward.append(sum(tourist.reward))
            static_reward.append(sum(tourist.staticReward))

        g=gini(reward)
        s_g=gini(static_reward)
        r_sum=sum(reward)
        s_r_sum=sum(static_reward)
        # print("-"*100)
        # print("Total Reward: ", r_sum, " Gini: ", g)
        # out_reward=[round(r, 1) for r in reward]
        # print("rewards: ", out_reward)
        # print("-"*100)
        self.env.close()
        # return self.env.spots, self.env.tourists
        return r_sum, s_r_sum, g, s_g

    def plot(self, history):
        print(history)

    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')