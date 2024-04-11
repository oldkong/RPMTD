# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import tensorflow as tf

from collections import deque

from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow
from drl_base_v7 import DRLBase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import activations
import argparse
import time
from datetime import datetime

# devices=tensorflow.config.experimental.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(devices[0], True)

class DQN(DRLBase):
    """Deep Q-Learning.
    """
    def __init__(self, env=None):
        super(DQN, self).__init__(env)

        self.model = self.build_model(num_action=len(self.env.spots)+1)

        # experience replay.
        self.memory_buffer = deque(maxlen=2000)
        # discount rate for q value.
        self.gamma = 0.99
        # epsilon of ε-greedy.
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01

    def load(self, filename='model/dqn.h5'):
        if os.path.exists(filename):
            self.model.load_weights(filename)
         
    def build_model(self, num_action=21):
        """basic model.
        """
        model = Sequential()
        model.add(Dense(2*num_action, activation=activations.relu, input_shape=(num_action,)))
        model.add(Dense(num_action, activation=activations.relu))
        model.add(Dense(num_action, activation=activations.softmax))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=tensorflow.keras.losses.CategoricalCrossentropy())

        return model

    def egreedy_action(self, state):
        """ε-greedy
        Arguments:
            state: observation
        Returns:
            action: action
        """
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(self.env.spots))
        else:
            reshape_state=np.reshape(state, (1, -1))
            q_values = self.model.predict(reshape_state)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.
        Arguments:
            state: observation
            action: action
            reward: reward
            next_state: next_observation
            done: if game done.
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """update epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """process batch data
        Arguments:
            batch: batch size
        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
        # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        # Q_target。
        # x[0], action, reward, observation, done
        states = np.array([d[0] for d in data]) # batch x 73
        next_states = np.array([d[3] for d in data])
        states = np.asarray(states).astype(np.float32)
        next_states = np.asarray(next_states).astype(np.float32)

        y = self.model.predict(states) # batch x 73， 对每个动作打分
        q = self.model.predict(next_states) 

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i]) # 对不对？！
            y[i][action] = target

        return states, y

    def train(self, episode, batch, model_name):
        """training 
        Arguments:
            episode: game episode
            batch： batch size
        Returns:
            history: training history
        """
        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation, touristId = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False
            allComplete=1
            for tourist in self.env.tourists:
                allComplete *= tourist.complete
            while allComplete==0 and touristId !=-1 :
                pervious_obser=observation
                action = self.egreedy_action(observation)
                observation, reward, done, touristId = self.env.step([touristId, action])
                reward_sum += reward
                self.remember(pervious_obser, action, reward, observation, done) #state, action, reward, next_state, done
                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)
                    count += 1
                    # reduce epsilon pure batch.
                    self.update_epsilon()
                allComplete=1 #检查所有旅客的状态
                for tourist in self.env.tourists:
                    allComplete *= tourist.complete

            if i % 1 == 0:
                print("-"*100)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("self.env.spots: ")
                for spot in self.env.spots:
                    print(spot)
                print("self.env.tourists: ")
                for tourist in self.env.tourists:
                    print(tourist)
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))
                

        self.model.save_weights('model/'+model_name+'.h5')
        return history

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default="dqn")
    parser.add_argument('--epoch', default="200")
    parser.add_argument('--batch_size', default="256")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    with tf.device('GPU:1'):
        ts = time.time()
        now = datetime.now()
        timestamp=now.strftime('%Y.%m.%d_%H_%M')
        dir="visual"
        model = DQN()

        args=parse_args()
        model_name=args.model_name
        epoch=int(args.epoch)
        batch_size=int(args.batch_size)
        print("args: ", args)
        model_name=model_name+"_"+str(epoch)+"_"+timestamp

        history = model.train(epoch, batch_size, model_name)
        model.save_history(history, model_name+".csv")
        print("-"*100, "\n", "-"*100, "\n", "-"*100, "\n")
        model.load(filename='model/'+model_name+'.h5')
        model.play()
        te = time.time()
        print("RUN TIME(s): ", te-ts)
