# from re import I, M
# import warnings
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from timer_v6 import Timer_v6
import math
import random
from datetime import datetime
import pandas as pd
from geopy import distance
import copy

from os.path import dirname, abspath
# project_dir = dirname(dirname(abspath(__file__)))
project_dir = dirname(abspath(__file__))
time_table = pd.read_csv(project_dir+'/data_/mix_duration_matrix_AOI.csv').to_numpy()[:,1:]

def convert_num_time(num):
    num = str(num)
    h,m=num.split('.')
    if m == '0':
        _m = ':00'
    elif m == '25':
        _m = ':15'
    elif m == '5':
        _m = ':30'
    elif m == '75':
        _m = ':45'
    time = h + _m
    return time

# 91个景点对应的名称： https://github.com/Ma-Lab-Public/Maekawa_Kyoto_POI_dataset/blob/main/AOI_data.csv

class Spot:
    def __init__(self, spotId, location, cMax=0, rMax=0.0, timeCost=0.0, alpha=1.2, rMin=-1.0, num=0, name='spot'):
        self.name = name
        # 景点编号
        self.spotId=spotId
        # 景点坐标,是一个含有两个值的tuple:(x, y)
        self.location=location
        # 景点推荐的游客容量
        self.cMax=cMax
        # 游客游览景点可以得到的最大回报
        self.rMax=rMax
        self.rMax_=rMax
        # 最小回报
        # 如果不限制负回报的大小，应该也不影响训练，因为负回报越大，越能规范智能体的行为
        # 只不过有可能测试模型的时候回报没那么好看
        self.rMin=rMin
        # 回报为0时景点的游客数量与推荐游客数量cMax的比值
        # 现在的回报函数用不上：r=max(math.cos((self.num/self.cMax)*(0.5*math.pi))*self.rMax, self.rMin)
        self.alpha=alpha
        # 游客游览景点所花费的时间
        self.timeCost=timeCost
        # 初始化游览该景点的人数为早上9点（36）
        self.num=num
        # 每个景点被浏览的次数
        self.visit_num = 0
    
    def reward(self):
        # 逻辑上有个问题就是 num 快到 cMax 3 倍附近的时候，r 就又有可能变成正的
        # 但是在训练、测试中应该不会有问题
        r=max((1-self.num/self.cMax)*self.rMax, self.rMin)
        return r

    def __str__(self):
        return "spotId: "+ str(self.spotId)+ ", location: "+ str(self.location)+ ", cMax: "+ str(self.cMax)+ ", rMax: "+ str(self.rMax)+ ", timeCost: "+ str(self.timeCost)+ ", alpha: "+ str(self.alpha)+ ", num: "+ str(self.num)

class Tourist:
    def __init__(self, touristId, timer: Timer_v6, startSpot, endSpot, speed=1.0, loc=[0, 0]):
        # 游客编号
        self.touristId = touristId
        # 游客总剩余时间计时器
        self.timer = timer
        # 起始景点的编号
        self.startSpot = startSpot
        # 终止景点的编号
        self.endSpot = endSpot
        # 游客有看过的景点的有序列表，最后一个是游客当前游览的景点或者刚刚游览完的景点
        self.route = []
        self.route_s = []         # route_s 不包含等待
        self.route_name = [] # 直接显示景点名字
        # 游客的移动速度
        self.speed = speed
        # 游客截至目前所获得的所有奖励
        # self.reward = 0.0
        self.reward = []
        self.congestion_reward = []
        self.staticReward = []  
        # 正在规划的为0，完不成的话置为-1，顺利完成的话置为1
        self.complete = 0
        # mask 掉去过的景点
        self.mask = [1 for i in range(92)]
        # info 用来记录每个reward对应的spot人数比
        self.info_before = []
        self.info_after = []
        # 游客的起始位置
        # self.start_point=(loc[0]+np.random.normal(0, 1), loc[1]+np.random.normal(0, 1))
        self.start_point=(loc[0], loc[1])
        self.unfrozen_time_step = 36
        self.frozen_time_step = 36
        self.frozen_duration = 0
        self.remain_time = 32
        # 记录游客访问每个景点的时间
        self.StartTimetoVisit = []
        
    
    def act(self, env, action, reward, commuteTime):
        # 把景点编号添加到路径里面
        assert action == env.spots[action].spotId
        self.route.append(env.spots[action].spotId)
        # self.realRoute.append(env.spots[action].spotId)
        # 对应spot的 visit_num 加 1
        env.spots[action].visit_num += 1
        # 对应景点的游览人数+1
        env.spots[action].num+=1
        # 累加游客游览景点获得的回报
        self.reward.append(reward)
        # 设置游览景点的计时器
        self.timer.useTime(timeCost=(commuteTime+env.spots[action].timeCost))
        if self.timer.getElapsed()<0:
            self.complete=1
        return True
    
    def __str__(self):
        return "Tourist {}: start_point: {}, startSpot: {}, endSpot: {},\n realRoute: {},\n Reward: {},\n staticReward: {}, elapsed: {}, nextFinish: {}, complete: {}.\n".format(self.touristId, self.start_point, self.startSpot, self.endSpot, self.realRoute, self.reward, self.staticReward, self.timer.getElapsed(), self.timer.nextFinish, self.complete) 

spotStatistic=[[[], []] for i in range(7)]
def generateData(numt=200):
    spotStatisticFile=open(project_dir+"/data_/statistic_v3.txt", "r")
    for l in spotStatisticFile.readlines():
        ls=l.strip().split("\t")
        id=int(ls[0])
        cate=0 if ls[2]=="num" else 1
        value=ls[3].strip("[").strip("]").split(",")
        if cate == 0:
            value=[int(v) for v in value]
        else:
            value=[float(v) for v in value]
        spotStatistic[id][cate]=value
    
    aoi_file=open(project_dir+"/data_/AOI_data.csv")
    spots=[]
    loc=[0, 0]
    for l in aoi_file.readlines():
        items=l.split(",")
        loc[0]+=float(items[1])
        loc[1]+=float(items[2])
        # spots.append(Spot(spotId=int(items[0]), location=(float(items[1]), float(items[2])), cMax=2*spotStatistic[0][0][int(items[0])], rMax=spotStatistic[0][1][int(items[0])], timeCost=0.4, alpha=1.2, num=spotStatistic[0][0][int(items[0])]))
        spots.append(Spot(spotId=int(items[0]), location=(float(items[1]), float(items[2])), cMax=2*spotStatistic[2][0][int(items[0])], rMax=spotStatistic[2][1][int(items[0])], timeCost=0.4, alpha=1.2, num=spotStatistic[2][0][int(items[0])]))
        rmin=0
        for s in spots:
            rmin+=s.rMax/2.0
        rmin/=len(spots)
        for s in spots:
            s.rMin=-rmin
    loc[0]/=len(spots)
    loc[1]/=len(spots)
    tourists=[]
    # s=random.randint(0, len(spots)-1)
    # e=s
    # while e==s:
    #     e=random.randint(0, len(spots)-1)    
    s=10
    e=21
    for i in range(len(tourists), numt):
        # s=random.randint(0, len(spots)-1)
        # e=s
        # while e==s:
        #     e=random.randint(0, len(spots)-1)
        # timer=Timer_v6(elapsed=3+2*random.random()), speed=random.uniform(5, 10)
        tourists.append(Tourist(touristId=i, timer=Timer_v6(elapsed=5), startSpot=s, endSpot=e, speed=10, loc=loc))
    print("LEN TOURISTS::: ", len(tourists))
    return spots, tourists


def generateDataSim(population_table, numt=200):
    spotStatisticFile=open(project_dir+"/data_/statistic_v3.txt", "r")
    for l in spotStatisticFile.readlines():
        ls=l.strip().split("\t")
        id=int(ls[0])
        cate=0 if ls[2]=="num" else 1
        value=ls[3].strip("[").strip("]").split(",")
        if cate == 0:
            value=[int(v) for v in value]
        else:
            value=[float(v) for v in value]
        spotStatistic[id][cate]=value
    
    aoi_file=open(project_dir+"/data_/AOI_data.csv")
    spots=[]
    loc=[0, 0]
    for l in aoi_file.readlines():
        items=l.split(",")
        loc[0]+=float(items[1])
        loc[1]+=float(items[2])
        # spots.append(Spot(spotId=int(items[0]), location=(float(items[1]), float(items[2])), cMax=2*spotStatistic[0][0][int(items[0])], rMax=spotStatistic[0][1][int(items[0])], timeCost=0.4, alpha=1.2, num=spotStatistic[0][0][int(items[0])]))
        spots.append(Spot(spotId=int(items[0]), location=(float(items[1]), float(items[2])), cMax=2*spotStatistic[2][0][int(items[0])], rMax=spotStatistic[2][1][int(items[0])], timeCost=0.4, alpha=1.2, num=spotStatistic[2][0][int(items[0])]))
        rmin=0
        for s in spots:
            rmin+=s.rMax/2.0
        rmin/=len(spots)
        for s in spots:
            s.rMin=-rmin
    loc[0]/=len(spots)
    loc[1]/=len(spots)
    # print("LOC: ", loc)
    tourists=[]
    s=10 #指定10点为起点
    e=21 #指定21点为终点
    sSim={}
    eSim={}
    for spot in spots: #算每个点到指定点（10,21）的距离
        sDist = math.hypot(spot.location[0] - spots[s].location[0], spot.location[1] - spots[s].location[1])
        eDist = math.hypot(spot.location[0] - spots[e].location[0], spot.location[1] - spots[e].location[1])
        sSim[sDist]=spot.spotId
        eSim[eDist]=spot.spotId
    skey=sorted(sSim)
    ekey=sorted(eSim)
    ss=[sSim[skey[0]], sSim[skey[1]], sSim[skey[2]]] #ss为起点的集合，集合里的3个点相邻很近
    ss = [9,10,11,54,55,56,65,66]
    es=[eSim[ekey[0]], eSim[ekey[1]], eSim[ekey[2]]] #es为终点的集合。。。
    es = [22,67,70,50]
    # print("START SPOTS: ", ss)
    # print("END SPOTS: ", es)
    for i in range(len(tourists), numt):
        tourists.append(Tourist(touristId=i, timer=Timer_v6(elapsed=5), startSpot=random.choice(ss), endSpot=random.choice(es), speed=10, loc=loc))

    return spots, tourists    


def generateDataSim_91Spots(population_table, numt=1):
    spots_info = pd.read_csv(project_dir+'/data_/91spots_info.csv')
    spots=[]

    for i in range(len(spots_info)):
        loc = (spots_info.iloc[i].loc1, spots_info.iloc[i].loc2)
        name = spots_info.iloc[i].aoi
        cMax = 0
        rMax = spots_info.iloc[i].r
        timeCost = spots_info.iloc[i].time/60
        rMin = 0
        num = 0
        spots.append(Spot(spotId = i, location = loc, cMax = cMax, rMax = rMax, timeCost = timeCost, rMin = rMin, num = num, name = name))

    tourists=[]
    s=10 #指定10点为起点
    e=21 #指定21点为终点
    sSim={}
    eSim={}
    for spot in spots: #算每个点到指定点（10,21）的距离
        sDist = math.hypot(spot.location[0] - spots[s].location[0], spot.location[1] - spots[s].location[1])
        eDist = math.hypot(spot.location[0] - spots[e].location[0], spot.location[1] - spots[e].location[1])
        sSim[sDist]=spot.spotId
        eSim[eDist]=spot.spotId
    skey=sorted(sSim)
    ekey=sorted(eSim)
    ss=[sSim[skey[0]], sSim[skey[1]], sSim[skey[2]]] #ss为起点的集合，集合里的3个点相邻很近
    ss = [9,10,11,54,55,56,65,66]
    es=[eSim[ekey[0]], eSim[ekey[1]], eSim[ekey[2]]] #es为终点的集合。。。
    es = [22,67,70,50]
    # print("START SPOTS: ", ss)
    # print("END SPOTS: ", es)
    for i in range(len(tourists), numt):
        tourists.append(Tourist(touristId=i, timer=Timer_v6(elapsed=5), startSpot=random.choice(ss), endSpot=random.choice(es), speed=10, loc=loc))

    return spots, tourists    


def interpolation_double(data):
    shape = data.shape
    data_double = np.ones((shape[0],shape[1]*2))

    for i in range(data_double.shape[1]):
        if i%2 ==0:
            data_double[:,i] = data[:,int(i/2)]
        else:
            data_double[:,i] = (data[:,int((i+1)%data_double.shape[-1]/2)] + data[:,int((i-1)/2)])/2
    return data_double.astype(int) 
            

def expedn48to96(path):
    b = pd.read_csv(path).to_numpy()[:,1].reshape(-1,232).transpose()
    assert b.shape == (232,48)
    a = interpolation_double(b)
    return a.astype(int) 


def get_Flickerdata_1H():
    d = pd.read_csv(project_dir+'/data_/1h_count_with_interpolation.csv').to_numpy()
    map = pd.read_csv(project_dir+'/data_/map72to76.csv').to_numpy()[:,1]

    p_data = np.ones([72,25])
    for i in range(72):
        p_data[i] = d[map[i]]

    return p_data[:,1:].astype(int) # shape 为 72x4


def get_Yahoo_population_data(Yahoo_data_path): # 获得yahoo人流数据
    # p_data = pd.read_csv('/home/y.kong/kyt_new/KYT_Route_recommendation/data/people_overtime_96_data.csv').to_numpy()[:,1:]
    p_data = expedn48to96(Yahoo_data_path) # yikun 的人流数据
    p_data_kwk = np.ones((232,96))
    map = pd.read_csv(project_dir+'/data_/72_91_spotID_mapping.csv').to_numpy()[:,1]
    population_data = np.ones((72,96))
    map91to91 = pd.read_csv(project_dir+'/data_/91to91_mapping.csv').to_numpy()[:,2]

    #### 将yikun的数据 map 到 maekawa的spot上面
    for i in range(len(map91to91)):
        p_data_kwk[i] = p_data[map91to91[i]]

    #### reset 祇王寺 with 愛宕念仏寺
    p_data_kwk[5] = p_data_kwk[3]

    #### reset 晴明神社 with 白峯神宮
    p_data_kwk[39] = p_data_kwk[38]
 
    #### reset 京都水族館 with 京都塔+東寺
    p_data_kwk[80] = ((p_data_kwk[81] + p_data_kwk[78])/2).astype(int)


    #### 将maekawa的数据 map 到 KONG WK 的spot上面
    for i in range(72):
        population_data[i] = p_data_kwk[map[i]]
    
    return population_data.astype(int) 

def get_Yahoo_population_data_91spots(Yahoo_data_path): # 获得yahoo人流数据
    # p_data = pd.read_csv('/home/y.kong/kyt_new/KYT_Route_recommendation/data/people_overtime_96_data.csv').to_numpy()[:,1:]
    p_data = expedn48to96(Yahoo_data_path) # yikun 的人流数据
    p_data_kwk = np.ones((232,96))
    map = pd.read_csv(project_dir+'/data_/72_91_spotID_mapping.csv').to_numpy()[:,1]
    population_data = np.ones((72,96))
    map91to91 = pd.read_csv(project_dir+'/data_/91to91_mapping.csv').to_numpy()[:,2]

    #### 将yikun的数据 map 到 maekawa的spot上面
    for i in range(len(map91to91)):
        p_data_kwk[i] = p_data[map91to91[i]]

    #### reset 祇王寺 with 愛宕念仏寺
    p_data_kwk[5] = p_data_kwk[3]

    #### reset 晴明神社 with 白峯神宮
    p_data_kwk[39] = p_data_kwk[38]
 
    #### reset 京都水族館 with 京都塔+東寺
    p_data_kwk[80] = ((p_data_kwk[81] + p_data_kwk[78])/2).astype(int)


    #### 将maekawa的数据 map 到 KONG WK 的spot上面
    for i in range(72):
        population_data[i] = p_data_kwk[map[i]]
    
    return population_data.astype(int) 


def updateSpotStatistic(index, env):
    # print("index: ", index)
    # diff=[spotStatistic[index][0][i]-spotStatistic[index-1][0][i] for i in range(len(spotStatistic[index][0]))]
    # new_rMax=spotStatistic[index][1]
    # for i in range(len(env.spots)):
    #     env.spots[i].num+=diff[i]
    #     env.spots[i].rMax=new_rMax[i]
    print("Didn't update", index, env)
        

class RREnv(Env):
    def __init__(self, use_Yahoo_population_data = False):
        # 得到景点人数随时间变化的table
        self.population_table  = np.ones((91,96))
        # spots, tourists=generateData()
        spots, tourists=generateDataSim(self.population_table)
        # 可能的动作：什么也不干、去第n个景点，
        # 什么也不干有可能是在游览，有可能是所有景点都满了，也有可能是剩余时间不够游览完一个景点了
        # 其实也可以将什么也不干定义成去现在所在的景点，但是，不是那么讲得通
        self.action_space = Discrete(len(spots)+1)
        # 将观察空间定义成回报与(通勤时间+游览时间)之比
        # 目前还不清楚回报时间之比的范围是多少暂且为 [-1000， 1000]
        self.observation_space = Box(low=np.array([np.float32(-1000) for i in range(len(spots)+1) ]), high=np.array([np.float32(1000) for i in range(len(spots)+1)]), dtype=np.float32)
        self.spots=spots
        self.tourists=tourists
        # 为了避免游客指定的起、止景点拥挤
        vipSpots=set()
        for tourist in self.tourists:
            vipSpots.add(tourist.startSpot)
            vipSpots.add(tourist.endSpot)
        self.vipSpots=vipSpots
        self.epoch=0
        now = datetime.now()
        timestamp=now.strftime('%Y.%m.%d_%H_%M')
        # self.log = open("visual/dynamicReward_"+timestamp+"_spotStatistic.log", "w")
        start_time = 9
        self.current_time  = 4 * start_time # 36 是早上9点

        # 用 population_table 初始化每个景点早上9点的人数(36 为 早上9点)
        # 更新 sopt 的 capacity
        if use_Yahoo_population_data == True:
            for i, s in enumerate(self.spots):
                s.num = self.population_table[s.spotId][36]
                s.cMax = self.population_table.max(axis = 1)[i] if self.population_table.max(axis = 1)[i]!=0 else 328

        for t in self.tourists:
            t.speed = 15



    # ====================================================================
    # 为给定的 tourist 和 spot 计算 reward
    # 通过为观察和 reward 使用一个计算方法来保证观察和 reward 之间的相关性
    # ====================================================================
    def calculateReward(self, touristId, action, t=1.0):
        # ====================================================================
        # tourist: self.tourists[touristId]
        # if action == len(self.spots): no spot
        # else: spot: self.spots[spotId]
        # ====================================================================
        reward, commuteTime=0.0, 0.0
        if action==len(self.spots):
            # self.tourists[touristId].realRoute.append(action)
            # self.tourists[touristId].reward.append(0)
            if len(self.tourists[touristId].route_s) != 0:
                spot = self.tourists[touristId].route_s[-1]    
            else: spot = self.tourists[touristId].startSpot        
            if (not self.checkCompleteAvailability(touristId, spot)) or (not self.checkAvailabilityAct(touristId, spot)) : # 等下去就没时间。或者等下去去不了终点
                reward= 0 
            else: reward = 0
            return False, reward, commuteTime

        else:
            # 如果能去这个景点
            if self.checkAvailabilityAct(touristId, action):
                # commuteTime=self.calculateDistance(touristId, self.spots[action])/self.tourists[touristId].speed
                commuteTime=self.calculateTime(touristId,action)
                r=self.spots[action].reward()
                reward=r
                # 如果去了这个景点还能去终点
                if self.checkCompleteAvailability(touristId, action):
                    # 如果还未开始游览
                    if len(self.tourists[touristId].route_s) == 0:
                        if action == self.tourists[touristId].startSpot:
                            reward = 1*reward if reward>=0 else reward
                        else:
                            reward = 1*reward if reward>=0 else reward
                    # 如果已经开始游览
                    else:
                        # 如果去过这个景点了
                        if action in self.tourists[touristId].route_s[:-1]:
                            reward = 0*reward if reward>=0 else 0*reward
                            # reward = -20
                        # 还没去过
                        else:
                            # 是终点
                            if action == self.tourists[touristId].endSpot:
                                reward = 1*reward if reward>=0 else -1*reward
                            # 不是终点
                            else:
                                # 是其他游客的起点或终点
                                if action != self.tourists[touristId].startSpot and action != self.tourists[touristId].endSpot and action in self.vipSpots:
                                    reward = 1*reward if reward>=0 else reward
                                # 不是其他游客的起点或终点
                                else:
                                    reward = reward if reward>=0 else reward
                    return True, reward, commuteTime
                
                else: # 如果去了这个景点就不能去终点了
                    if action == self.tourists[touristId].endSpot:
                        if len(self.tourists[touristId].route_s)!=0: 
                            reward = 2*reward if reward>=0 else -2*reward
                            return True, reward, commuteTime
                        else:
                            reward = -2*reward if reward>=0 else 2*reward
                            return True, reward, commuteTime
                    else:
                        reward = -2*reward if reward>=0 else 2*reward
                        return False, reward, commuteTime
            # 去不了，什么具体动作也没有
            else:
                # 给个负奖励
                reward = -2*self.spots[action].rMax
                # 时间-1应该也不会让模型从中学到任何东西，暂时去掉
                # self.tourists[touristId].realRoute.append(-1)
                return False, reward, commuteTime

    def calculateStaticReward(self, touristId, action, t=1.0):

        staticReward=self.spots[action].rMax
        route=self.tourists[touristId].route[0:-1]
        # 如果还未开始游览
        if len(route) == 0:
            if action == self.tourists[touristId].startSpot:
                staticReward = 2*staticReward
            else:
                staticReward = -2*staticReward
        # 如果已经开始游览
        else:
            # 如果去过这个景点了
            if action in route:
                staticReward = -2*staticReward
            # 还没去过
            else:
                # 是终点
                if action == self.tourists[touristId].endSpot:
                    staticReward = 2*staticReward
                else:
                    staticReward = 1*staticReward
        return staticReward

    # ====================================================================
    # 主动作函数
    # ====================================================================
    def step(self, action):
        self.epoch+=1

        if self.current_time % 1 ==0:
            self.spot_attend_ratio.append(np.var([s.num/s.cMax for s in self.spots]))
            self.spot_attend_ratio_detail.append([s.num/s.cMax for s in self.spots])

        if self.epoch%len(self.spots)==0:
            # # 顺序循环使用历史统计数据
            # updateSpotStatistic(int(self.epoch/len(self.spots)%len(spotStatistic)), self)
            # 随机使用历史统计数据
            updateSpotStatistic(random.randint(0, len(spotStatistic)-1), self)
            # print("*"*100)
            # self.log.write("*"*100+"\n")
            # self.log.write("*"*100+"\n")
            # print("Updated spots's statistic information")
            # self.log.write("Updated spots's statistic information")
            # print("*"*100)
            # print("*"*100)
            # self.log.write("*"*100+"\n")
            # self.log.write("*"*100+"\n")
            # for spot in self.spots:
            #     print(spot)
            #     self.log.write(spot.__str__()+"\n")
            # print("*"*100)
            # print("*"*100)
            # self.log.write("*"*100+"\n")
            # self.log.write("*"*100+"\n")

        # touristId, action =action[0], action[1]
        # r_overtime=[]
        # reward=0.0
        # done = False
        # nextTourist=-1        

        # ====================================================================
        # 执行动作
        # ====================================================================
        # 根据action，更新 population 人数，tourist 状态
        for i in range(len(action)):
            a = action[i]
            if a == None:
                continue
            else:
                if a == len(self.spots): # 如果是原地等待

                    ############# 更新游客信息 ################
                    if self.tourists[i].frozen_duration != 0: # 如果已经开始游玩，才需要从之前的spot退出
                        previous_spot = self.tourists[i].route[-1] 
                        if previous_spot != (len(self.spots)): # 如果之前在spot，而不是等待
                            self.population_table[previous_spot][self.tourists[i].frozen_time_step: self.tourists[i].unfrozen_time_step] -= 1 # 退出之前的spot, 更新 population 信息

                    self.tourists[i].frozen_time_step = self.tourists[i].unfrozen_time_step
                    self.tourists[i].unfrozen_time_step += 1
                    self.frozen_duration = 1

                    self.tourists[i].route.append(a) # 更新游客游览记录
                    # if a != len(self.spots)+1:
                    #     self.tourists[i].route_s.append(a)
                    self.tourists[i].remain_time -= 1

                    if self.tourists[i].remain_time <= 0 or self.tourists[i].route_s[-1] == self.tourists[i].endSpot:
                        self.tourists[i].complete = 1

                else: # 如果是去某个spot
                    # commuteTime = self.calculateDistance(i, self.spots[a]) / self.tourists[i].speed # 单位是小时
                    commuteTime = self.calculateTime(i,a) # 单位是小时
                    total_time_step = round((self.spots[a].timeCost + commuteTime)*4)
                    
                    ############# 更新游客信息 ################
                    if self.tourists[i].frozen_duration != 0: # 如果已经开始游玩
                        previous_spot = self.tourists[i].route[-1] 
                        if previous_spot != (len(self.spots)): # 如果在spot而不是等待
                            self.population_table[previous_spot][self.tourists[i].frozen_time_step: self.tourists[i].unfrozen_time_step] -= 1 # 退出之前的 spot，更新 population 数据
                    
                    self.tourists[i].frozen_time_step = self.tourists[i].unfrozen_time_step
                    self.tourists[i].unfrozen_time_step += total_time_step
                    self.tourists[i].frozen_duration = total_time_step

                    self.population_table[a][self.tourists[i].frozen_time_step: self.tourists[i].unfrozen_time_step] += 1 # 参加新的 spot， 更新 population 数据
                    
                    self.tourists[i].route.append(a) # 更新游客游览记录
                    self.tourists[i].route_name.append(self.spots[a].name)
                    self.tourists[i].StartTimetoVisit.append(convert_num_time(self.current_time/4))
                    self.tourists[i].route_s.append(a)
                    self.tourists[i].mask[a] = 0
                    self.tourists[i].remain_time -= total_time_step
                    
                    if self.tourists[i].remain_time <= 0 or self.tourists[i].route_s[-1] == self.tourists[i].endSpot:
                        self.tourists[i].complete = 1
                    
                    ############# 更新spot信息 ################
                    self.spots[a].visit_num += 1 # 更新spot的游览记录

        
        ################## 更新spot里的人数 ################
        for i in range(len(self.spots)):
            self.spots[i].num = self.population_table[i][self.current_time]

            
        ################### 计算reward, reward per time 以及 action所对应的spot人数比 #####################
        reward_list = []
        congestion_reward_list = []
        rpt_ist = []
        for i in range(len(action)):
            a = action[i]
            if a == None:
                r = None
                rpt = None
                spot_ratio = None
                congestion_reward = None

            else:
                _, r, _ = self.calculateReward(i,a)
                congestion_reward = max(r,0)
                if a == len(self.spots):
                    rpt = 0
                else:
                    previous_loc = self.spots[self.tourists[i].route_s[-2]].location if len(self.tourists[i].route_s) >= 2 else self.tourists[i].start_point
                    current_loc = self.spots[a].location
                    # rpt = r/(self.dis_s2s(previous_loc, current_loc)/self.tourists[i].speed + self.spots[a].timeCost) if r >= 0 else r
                    rpt = r/(self.calculateTime(i, a) + self.spots[a].timeCost) if r >= 0 else r
                
                spot_ratio = self.spots[a].num/self.spots[a].cMax if a != len(self.spots) else None

                spots_attend_ratio_list = [s.num/s.cMax for s in self.spots]
                var = np.var(spots_attend_ratio_list)

                consider_fairness_global = True
                if consider_fairness_global:
                    r += 1/var  # 0.1为超参数，需要后期调整。         

            self.tourists[i].info_after.append(spot_ratio)
            reward_list.append(r)   
            rpt_ist.append(rpt)     
            congestion_reward_list.append(congestion_reward)     
             

        ################## 生成 next_state ###################
        # self.current_time += 1 # 在生成新state的时候才对 current_time 加 1
        self.current_time = min([t.unfrozen_time_step for t in self.tourists])

        for i in range(len(self.spots)): # 再次更新spot里的人数
            self.spots[i].num = self.population_table[i][self.current_time]

        
        ########## 更新spot的分数 ###############
        if self.local_fairness:
            if self.current_time > self.tigger_time:
                for s in self.spots:
                    if s.visit_num == 0:
                        s.rMax = 10 * s.rMax_
                    elif s.visit_num == 1:
                        s.rMax = 2.5 * s.rMax_
                    else: 
                        s.rMax = 1 * s.rMax_

        spot_state = np.ones((91,4))

        for i in range(91):
            spot_state[i][0] = self.spots[i].cMax
            spot_state[i][1] = self.spots[i].rMax
            spot_state[i][2] = self.spots[i].timeCost*4
            spot_state[i][3] = self.spots[i].num



        ############# done_list ###############
        done_list = []
        for t in self.tourists:
            done_list.append(t.complete)

        
        return spot_state, rpt_ist, reward_list, done_list, congestion_reward_list



    
    def checkAvailabilityAct(self, touristId, action):
        if action == self.tourists[touristId].endSpot:
            if self.tourists[touristId].remain_time + round(4*self.spots[self.tourists[touristId].endSpot].timeCost) >= 0:
                return True
            else: return False
        else:     
            if self.tourists[touristId].remain_time >= 0:
                return True 
            else: return False 

    def checkCompleteAvailability(self, touristId, spotId):
        if spotId != self.tourists[touristId].endSpot:
            # commuteTime = round((self.calculateDistance(touristId, self.spots[self.tourists[touristId].endSpot]) / self.tourists[touristId].speed)*4)
            commuteTime = round(self.calculateTime(touristId, spotId))
            if commuteTime <= self.tourists[touristId].remain_time:
                return True
            else: return False
        else: 
            if self.tourists[touristId].remain_time + round(4*self.spots[self.tourists[touristId].endSpot].timeCost) >= 0:
                return True
            else: return False
    
    def calculateTime_id2id(self,id1,id2):
        # 输入 景点id1 景点id2
        # 输出 2个景点间所需时间（小时）
        # time_table = pd.read_csv(project_dir+'/data/mix_duration_matrix_AOI.csv').to_numpy()[:,1:]

        if id1 == id2:
            timeCost = 0
        else:
            timeCost = time_table[id1, id2] if time_table[id1, id2] > 0 else 900-time_table[id1, id2] #得到的单位为 秒

        timeCost = timeCost/3600 # 单位转换为 小时
        return timeCost


    def calculateTime(self, touristId, spotID):
        # 输入 游客ID， 景点ID
        # 输出 所需时间 （小时）
        # time_table = pd.read_csv(project_dir+'/data/mix_duration_matrix_AOI.csv').to_numpy()[:,1:]

        if len(self.tourists[touristId].route_s)==0:
            currentID=self.tourists[touristId].startSpot
        else:
            currentID=self.tourists[touristId].route_s[-1]

        if currentID == spotID:
            timeCost = 0
        else:
            timeCost = time_table[spotID, currentID] if time_table[spotID, currentID] > 0 else 900-time_table[spotID, currentID] #得到的单位为 秒
        
        timeCost = timeCost/3600 # 单位转换为 小时
        
        return timeCost

    def calculateDistance(self, touristId, s: Spot):
        if len(self.tourists[touristId].route_s)==0:
            pos=self.tourists[touristId].start_point
        else:
            pos=self.spots[self.tourists[touristId].route_s[-1]].location
        # 交换经纬度
        pos_gps = (pos[0],pos[1])
        s_gps = (s.location[0], s.location[1])
        return distance.distance(pos_gps, s_gps).km

    def dis_s2s(self, spot_1_loc, spot_2_loc):
        if spot_1_loc == spot_2_loc:
            return 0
        else: 
            spot_1_gps = (spot_1_loc[0], spot_1_loc[1])
            spot_2_gps = (spot_2_loc[0], spot_2_loc[1])
            return distance.distance(spot_1_gps, spot_2_gps).km

    # def process_mask(self, i_t):
    #     mask = copy.deepcopy(self.tourists[i_t].mask)
    #     # if self.tourists[i_t].remain_time > 2:
    #     #     mask[self.tourists[i_t].endSpot] = 0

    #     if len(self.tourists[i_t].route_s)>0:
    #         current_loc = self.spots[self.tourists[i_t].route_s[-1]].location
    #     else:
    #         current_loc = self.tourists[i_t].start_point

    #     end_loc = self.spots[self.tourists[i_t].endSpot].location

    #     for i in range(len(self.spots)):
    #         # 去sport的时间 + 游玩时间 + 从spot去终点时间 < remain_time
    #         i_loc = self.spots[i].location
    #         if round(4*(self.dis_s2s(i_loc, current_loc)/self.tourists[i_t].speed + self.spots[i].timeCost)) + round(4*(self.dis_s2s(i_loc, end_loc)/self.tourists[i_t].speed)) > self.tourists[i_t].remain_time - 1:
    #         # if 4*((self.dis_s2s(i_loc, current_loc) + self.dis_s2s(i_loc, end_loc))/self.tourists[i_t].speed + self.spots[i].timeCost) > self.tourists[i_t].remain_time-1:
    #             mask[i] = 0

    #     # 如果等下去就没有时间的话
    #     if (int(4*(self.dis_s2s(end_loc, current_loc)/self.tourists[i_t].speed)) + 1 +1) > self.tourists[i_t].remain_time:
    #         mask[-1] = 0

    #     if self.tourists[i_t].remain_time > 10: # 10 是一个超参数。宇治到市区要7，但是要不留一些buffer的话，旅客在宇治没法到终点。
    #         mask[self.tourists[i_t].endSpot] = 0
    #     else: 
    #         mask[self.tourists[i_t].endSpot] = 1

    #     return mask

    def process_mask(self, i_t):
        mask = copy.deepcopy(self.tourists[i_t].mask)
        # if self.tourists[i_t].remain_time > 2:
        #     mask[self.tourists[i_t].endSpot] = 0

        if len(self.tourists[i_t].route_s)>0:
            current_ID = self.tourists[i_t].route_s[-1]
        else:
            current_ID = self.tourists[i_t].startSpot

        end_ID = self.tourists[i_t].endSpot

        for i in range(len(self.spots)):
            # 去sport的时间 + 游玩时间 + 从spot去终点时间 < remain_time
            if round(4*(self.spots[i].timeCost+self.calculateTime_id2id(current_ID,i))) + round(4*self.calculateTime_id2id(i,end_ID)) > self.tourists[i_t].remain_time - 1:
                mask[i] = 0

        # 如果等下去就没有时间的话
        if (int(4*self.calculateTime_id2id(current_ID, end_ID)) + 2) > self.tourists[i_t].remain_time:
            mask[-1] = 0

        if self.tourists[i_t].remain_time > 10: # 10 是一个超参数。宇治到市区要7，如果不留一些buffer的话，旅客在宇治没法到终点。
            mask[self.tourists[i_t].endSpot] = 0
        else: 
            mask[self.tourists[i_t].endSpot] = 1

        return mask

    def reset(self, local_fairness = True, start_time = 9, startSpot = 0, endSpot = 59, budget = 8, personalize_score = False, personalize_score_path = project_dir+'/data_/personalize_spots_prob_01.csv', use_simu_flow = False, use_Yahoo_population_data = True, use_Flicker_data = False, tourist_start_time_diverse = False, sim_data_path = '/home/y.kong/kyt_new/KYT_Route_recommendation/data_/Yahoo_data_ave.csv'):
        # spots, tourists=generateData()
        # personalize_s = pd.read_csv(personalize_score_path,header=None).to_numpy()
        # normalization score to 0~10
        # self.personalize_score = (10*(personalize_s - np.min(personalize_s))/np.ptp(personalize_s)).astype(float)

        self.local_fairness = local_fairness
        self.population_table  = np.zeros((232,96))
        sim_data_path = project_dir+'/data_/sim_data_ave.csv'
        self.Yahoo_1H_96 = interpolation_double(pd.read_csv(sim_data_path).values[:,-1].reshape(-1,232).transpose())/16
        # self.Flicker_1H_96 = interpolation_double(interpolation_double(get_Flickerdata_1H()))
        # self.simu_flow = self.get_simulation_flow()

        if use_Yahoo_population_data:
            self.population_table += self.Yahoo_1H_96

        if use_Flicker_data:
            self.population_table += self.Flicker_1H_96

        if use_simu_flow:
            self.population_table += self.simu_flow

        self.population_table = self.population_table.astype(int)

        spots, tourists=generateDataSim_91Spots(self.population_table)
        self.spots = spots
        self.tourists = tourists
        self.current_time  = int(4 * start_time) # 36 是早上9点
        
        # 用yikun的打分
        if personalize_score:
            try: 
                personalize_s = pd.read_csv(personalize_score_path,header=None).to_numpy().reshape(91,1)
                # normalization score to 0~10
                self.personalize_score = (10*(personalize_s - np.min(personalize_s))/np.ptp(personalize_s)).astype(float)

                for i in range(len(self.spots)):
                    self.spots[i].rMax = self.personalize_score[i].item()
                    self.spots[i].rMax_ = self.spots[i].rMax
                    
            except FileNotFoundError as e:
                print('yikun score file not found, use default score',e)


        # 重新选择tourists的终点
        for t in self.tourists:
            t.endSpot = endSpot
            t.startSpot = startSpot
            t.remain_time = budget * 4
            t.start_point = self.spots[endSpot].location
            t.unfrozen_time_step = int(4 * start_time)
            
            
        # 用 population_table 初始化每个景点早上9点的人数(36 为 早上9点)
        for i, s in enumerate(self.spots):
            s.num = self.population_table[s.spotId][self.current_time]

        # 更新 sopt 的 capacity
        if use_Yahoo_population_data or use_Flicker_data:
            for i,s in enumerate(self.spots):
                s.cMax = max(self.population_table.max(axis = 1)[i], 5) # 每个景点至少可以容纳5个人
                
        if tourist_start_time_diverse:
            for t in self.tourists:
                t.unfrozen_time_step += random.randint(0,4)

        self.tigger_time = int(min([t.unfrozen_time_step for t in self.tourists]) + 0.5*(max([t.unfrozen_time_step+t.remain_time for t in self.tourists])-min([t.unfrozen_time_step for t in self.tourists])))

        spot_state = np.ones((len(self.spots),4))
        for i in range(len(self.spots)):
            spot_state[i][0] = self.spots[i].cMax
            spot_state[i][1] = self.spots[i].rMax 
            spot_state[i][2]= self.spots[i].timeCost*4
            spot_state[i][3] = self.spots[i].num  

        ########  每个景点不设负分 ##########
        for s in self.spots:
            s.rMin = 0

        self.spot_attend_ratio = []
        self.spot_attend_ratio_detail= []
        return spot_state 
    
    # def reset(self, use_simu_flow = False, use_Yahoo_population_data = True, use_Flicker_data = False, tourist_start_time_diverse = True, use_actual_timecost = True, Yahoo_data_path = project_dir+'/data/aid_count.csv'):
    #         # spots, tourists=generateData()
    #         self.population_table  = np.zeros((72,96))

    #         # 分别获得 Flicker 和 Yahoo 的数据
    #         self.Yahoo_1H_96 = get_Yahoo_population_data(Yahoo_data_path)
    #         self.Flicker_1H_96 = interpolation_double(interpolation_double(get_Flickerdata_1H()))
    #         self.simu_flow = self.get_simulation_flow()

    #         if use_Yahoo_population_data:
    #             self.population_table += self.Yahoo_1H_96

    #         if use_Flicker_data:
    #             self.population_table += self.Flicker_1H_96

    #         if use_simu_flow:
    #             self.population_table += self.simu_flow

    #         self.population_table = self.population_table.astype(int)

    #         spots, tourists=generateDataSim(self.population_table)
    #         self.spots = spots
    #         self.tourists = tourists
    #         start_time = 9
    #         self.current_time  = 4 * start_time # 36 是早上9点

    #         # 重新选择tourist的起始点 以及 startSpot
    #         for t in self.tourists:
    #             # t.startSpot = random.randint(0,71)
    #             t.startSpot = random.choice([9,10,11,70,66,67])

    #             while t.startSpot in [0,25,68,69] or t.startSpot == t.endSpot: #不去远的spot(延暦寺, 上賀茂神社, 醍醐寺, 平等院),startSpot不等于endSpot
    #                 t.startSpot = random.randint(0,71)
                
    #         for t in self.tourists:
    #             start_s = random.randint(0,71)
    #             while self.dis_s2s(self.spots[start_s].location, self.spots[t.startSpot].location) > 2:
    #                 start_s = random.randint(0,71)
    #             t.start_point = self.spots[start_s].location
                
    #         # 如果不用 population_data ， 将population 表格设为KONG WK版本的值
    #         if use_Yahoo_population_data == False and use_Flicker_data == False:
    #             for i in range(len(self.spots)):
    #                 self.population_table[i] = self.spots[i].num

    #         # 用 population_table 初始化每个景点早上9点的人数(36 为 早上9点)
    #         for i, s in enumerate(self.spots):
    #             s.num = self.population_table[s.spotId][36]

    #         # 更新 sopt 的 capacity
    #         if use_Yahoo_population_data or use_Flicker_data:
    #             for i,s in enumerate(self.spots):
    #                 s.cMax = max(self.population_table.max(axis = 1)[i], 5) # 每个景点至少可以容纳5个人
            
    #         # 更新 spot 的 游玩时间
    #         if use_actual_timecost:
    #             # map = pd.read_csv('/home/y.kong/kyt_new/KYT_Route_recommendation/data/72_91_spotID_mapping.csv').to_numpy()[:,1]
    #             # timecost = pd.read_csv('/home/y.kong/kyt_new/KYT_Route_recommendation/data/duration_test3.csv').to_numpy()[:,1]

    #             map = pd.read_csv(project_dir+'/data/72_91_spotID_mapping.csv').to_numpy()[:,1]
    #             timecost = pd.read_csv(project_dir+'/data/duration_test3.csv').to_numpy()[:,1]

    #             map = list(map)
    #             timecost = list(timecost)

    #             for i in range(len(self.spots)):
    #                 self.spots[i].timeCost = max(timecost[map[i]] / 60, 0.2)

    #         # 更新 spot score
    #         score_list = list(pd.read_csv(project_dir+'/data/spot_score.csv').iloc[:,3].to_numpy())
    #         for i, s in enumerate(self.spots):
    #             s.rMax = score_list[i]
    #             s.rMax_ = s.rMax


    #         if tourist_start_time_diverse:
    #             for t in self.tourists:
    #                 t.unfrozen_time_step += random.randint(0,2)

    #         spot_state = np.ones((72,4))
    #         for i in range(72):
    #             spot_state[i][0] = self.spots[i].cMax
    #             spot_state[i][1] = self.spots[i].rMax
    #             spot_state[i][2]= self.spots[i].timeCost
    #             spot_state[i][3] = self.spots[i].num

    #         self.spots[52].rMax = 1 # 霊山観音
    #         self.spots[35].timeCost = 0.5 # 三千院
    #         # 
    #         ########  每个景点不设负分 ##########
    #         for s in self.spots:
    #             s.rMin = 0

    #         self.spot_attend_ratio = []
    #         self.spot_attend_ratio_detail= []
    #         return spot_state 


    def get_simulation_flow(self):
       data = pd.read_csv('/home/y.kong/kyt_new/KYT_Route_recommendation/data_/new_aoi_boundary.csv')['boundary']
       loc = np.ones([232,4])

       for i,l in enumerate(data):
         l_ = l.split()
         loc[i][0] = float(l_[0].split('[')[1])
         loc[i][1] = float(l_[1])
         loc[i][2] = float(l_[2])
         loc[i][3] = float(l_[3].split(']')[0])
       
       trj_loc = pd.read_csv('/home/y.kong/kyt_new/KYT_Route_recommendation/data_/Hakusan_trj_data.csv').iloc[:,1:].to_numpy()

       for t in trj_loc:
        lng, lat = t[-2], t[-1]
        for i, lo in enumerate(loc):
            if lng > lo[0] and lng < lo[2] and lat > lo[1] and lat < lo[3]:
                t[-1] = i
                t[-2] = i
            continue

       trj_loc[:,1] = trj_loc[:,1]*4
       trj_loc[:,2] = trj_loc[:,2] //15
       trj_loc[:,3] = trj_loc[:,2] + trj_loc[:,1]

       simu_flow_data = np.zeros([232,96])

       for it in trj_loc:
        time = int(it[-2])
        spot = int(it[-1])
        simu_flow_data[spot][time] += 1 
        # 这个结果是基于这版排序 https://github.com/Ma-Lab-Public/POI_datasets/blob/main/Kyoto_POI_datasets/AOI_POI_GRID_v202205/new_aoi_boundary.csv
        
        map91to91 = pd.read_csv(project_dir+'/data_/91to91_mapping.csv').to_numpy()[:,2]
        map = pd.read_csv(project_dir+'/data_/72_91_spotID_mapping.csv').to_numpy()[:,1]
        inter = np.zeros([232,96])
        final = np.zeros([72,96])

        for i in range(91):
            inter[i] = simu_flow_data[map91to91[i]]

        for i in range(72):
            final[i] = inter[map[i]]

        return final








 

    

