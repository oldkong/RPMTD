import warnings
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from timer_v6_static_reward import Timer, SpotTimer, Timer_v6
import math
import random
from datetime import datetime

from os.path import dirname, abspath
project_dir = dirname(dirname(abspath(__file__)))


class Spot:
    def __init__(self, spotId, location, cMax=0, rMax=0.0, timeCost=0.0, alpha=1.2, rMin=-1.0, num=0):
        # 景点编号
        self.spotId=spotId
        # 景点坐标,是一个含有两个值的tuple:(x, y)
        self.location=location
        # 景点推荐的游客容量
        self.cMax=cMax
        # 游客游览景点可以得到的最大回报
        self.rMax=rMax
        # 最小回报
        # 如果不限制负回报的大小，应该也不影响训练，因为负回报越大，越能规范智能体的行为
        # 只不过有可能测试模型的时候回报没那么好看
        self.rMin=rMin
        # 回报为 0 时景点的游客数量与推荐游客数量cMax的比值
        # 在现在这个版本回报函数中没用到：r=max(math.cos((self.num/self.cMax)*(0.5*math.pi))*self.rMax, self.rMin  )
        self.alpha=alpha
        # 游客游览景点所花费的时间
        self.timeCost=timeCost
        # 当前时刻游览该景点的人数
        self.num=num
    
    def reward(self):
        # 逻辑上有个问题就是 num 快到 cMax 3 倍附近的时候，r 就又有可能变成正的
        # 但是在训练和测试中不会有问题
        # r=max(math.cos((self.num/self.cMax)*(0.5*math.pi))*self.rMax, self.rMin  )
        if self.num < 2.01*self.cMax:
            r=max(math.cos((self.num/self.cMax)*(0.5*math.pi))*self.rMax, self.rMin)
        else:
            r=self.rMin
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
        self.realRoute = []        
        # 游客的移动速度
        self.speed = speed
        # 游客截至目前所获得的所有奖励
        self.reward = []
        self.staticReward = []
        # 正在规划的为0，完不成的话置为-1，顺利完成的话置为1
        self.complete = 0
        # 游客的起始位置
        self.start_point=(loc[0]+np.random.normal(0, 1), loc[1]+np.random.normal(0, 1))
    
    def act(self, env, action, reward, commuteTime):
        # 游客没在游览景点
        # if self.spotTimer==0 or self.spotTimer._start is None:
        # 把景点编号添加到路径里面
        assert action == env.spots[action].spotId
        self.route.append(env.spots[action].spotId)
        self.realRoute.append(env.spots[action].spotId)
        # 对应景点的游览人数+1
        env.spots[action].num+=1
        # 累加游客游览景点获得的回报
        # self.reward+=reward
        self.reward.append(reward)
        # 设置游览景点的计时器
        self.timer.useTime(timeCost=(commuteTime+env.spots[action].timeCost))
        if self.timer.getElapsed()<0:
            self.complete=1
        return True
    
    def __str__(self):
        return "Tourist {}: start_point: {}, startSpot: {}, endSpot: {},\n realRoute: {},\n Reward: {},\n staticReward: {}, elapsed: {}, nextFinish: {}, complete: {}.\n".format(self.touristId, self.start_point, self.startSpot, self.endSpot, self.realRoute, self.reward, self.staticReward, self.timer.getElapsed(), self.timer.nextFinish, self.complete) 

spotStatistic=[[[], []] for i in range(7)]
def generateData(tNum=180):
    spotStatisticFile=open(project_dir+"/data/statistic_v3.txt", "r")
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
    
    aoi_file=open(project_dir+"/data/AOI_data.csv")
    spots=[]
    loc=[0, 0]
    for l in aoi_file.readlines():
        items=l.split(",")
        loc[0]+=float(items[1])
        loc[1]+=float(items[2])
        spots.append(Spot(spotId=int(items[0]), location=(float(items[1]), float(items[2])), cMax=2*spotStatistic[0][0][int(items[0])], rMax=spotStatistic[0][1][int(items[0])], timeCost=0.4, alpha=1.2, num=spotStatistic[0][0][int(items[0])]))
    loc[0]/=len(spots)
    loc[1]/=len(spots)
    # print("LOC: ", loc)
    tourists=[]
    for i in range(len(tourists), tNum):
        s=random.randint(0, len(spots)-1)
        e=s
        while e==s:
            e=random.randint(0, len(spots)-1)
        tourists.append(Tourist(touristId=i, timer=Timer_v6(elapsed=3), startSpot=s, endSpot=e, speed=random.uniform(5, 10), loc=loc))

    return spots, tourists

def updateSpotStatistic(index, env):
    diff=[spotStatistic[index][0][i]-spotStatistic[index-1][0][i] for i in range(len(spotStatistic[index][0]))]
    new_rMax=spotStatistic[index][1]
    for i in range(len(env.spots)):
        env.spots[i].num+=diff[i]
        env.spots[i].rMax=new_rMax[i]
        

class RREnv(Env):
    def __init__(self, spots=None, tourists=None):
        # spots, tourists=generateData()
        # 可能的动作：什么也不干、去第n个景点，
        # 什么也不干有可能是在游览，有可能是所有景点都满了，也有可能是剩余时间不够游览完一个景点了
        # 其实也可以将什么也不干定义成去现在所在的景点，但是，不是那么讲得通
        self.action_space = Discrete(len(spots)+1)
        # 将观察空间定义成回报与(通勤时间+游览时间)之比
        # 目前还不清楚回报时间之比的范围是多少暂且为 [-1000, 1000]
        self.observation_space = Box(low=np.array([np.float32(-1000) for i in range(len(spots)+1) ]), high=np.array([np.float32(1000) for i in range(len(spots)+1)]), dtype=np.float32)
        self.spots=spots
        self.tourists=tourists
        vipSpots=set()
        for tourist in self.tourists:
            vipSpots.add(tourist.startSpot)
            vipSpots.add(tourist.endSpot)
        self.vipSpots=vipSpots
        self.epoch=0
        now = datetime.now()
        timestamp=now.strftime('%Y.%m.%d_%H_%M')
        # self.log = open("visual/staticReward_"+timestamp+"_spotStatistic.log", "w")
        self.log = None

    def setTourists(self, tourists):
        self.tourists=tourists

    def setSpots(self, spots):
        self.spots=spots

    # ====================================================================
    # 为给定的 tourist 和 spot 计算 reward
    # 通过为观察和 reward 使用一个计算方法来保证观察和 reward 之间的相关性
    # ====================================================================
    def calculateReward_dynamic(self, touristId, action, t=1.0):
        # ====================================================================
        # tourist: self.tourists[touristId]
        # if action == len(self.spots): no spot
        # else: spot: self.spots[spotId]
        # ====================================================================
        reward, commuteTime=0.0, 0.0
        if action==len(self.spots):
            self.tourists[touristId].realRoute.append(action)
            self.tourists[touristId].reward.append(0)
            if not self.checkAvailabilityAct(touristId, self.tourists[touristId].endSpot):
                reward=-20
            return False, reward, commuteTime
        else:
            # 如果能去这个景点
            if self.checkAvailabilityAct(touristId, action):
                commuteTime=self.calculateDistance(touristId, self.spots[action])/self.tourists[touristId].speed
                r=self.spots[action].reward()
                reward=r
                # 如果去了这个景点还能去终点
                if self.checkCompleteAvailability(touristId, action):
                    # 如果还未开始游览
                    if len(self.tourists[touristId].route) == 0:
                        if action == self.tourists[touristId].startSpot:
                            reward = 2*reward if reward>=0 else reward
                        else:
                            reward = -2*reward if reward>=0 else 2*reward
                    # 如果已经开始游览
                    else:
                        # 如果去过这个景点了
                        if action in self.tourists[touristId].route:
                            reward = -2*reward if reward>=0 else 2*reward
                        # 还没去过
                        else:
                            # 是终点
                            if action == self.tourists[touristId].endSpot:
                                reward = 2*reward if reward>=0 else -2*reward
                            # 不是终点
                            else:
                                # 是其他游客的起点或终点
                                if action != self.tourists[touristId].startSpot and action != self.tourists[touristId].endSpot and action in self.vipSpots:
                                    reward = 0.8*reward if reward>=0 else reward
                                # 不是其他游客的起点或终点
                                else:
                                    reward = reward if reward>=0 else reward
                    # print("Y"*100)
                    return True, reward, commuteTime
                # 如果去了这个景点就不能去终点了
                else:
                    if action == self.tourists[touristId].endSpot:
                        if len(self.tourists[touristId].route)!=0: 
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


    def calculateReward_static(self, touristId, action, t=1.0):
        # ====================================================================
        # tourist: self.tourists[touristId]
        # if action == len(self.spots): no spot
        # else: spot: self.spots[spotId]
        # ====================================================================
        reward, commuteTime=0.0, 0.0
        if action==len(self.spots):
            self.tourists[touristId].realRoute.append(action)
            self.tourists[touristId].reward.append(0)
            if not self.checkAvailabilityAct(touristId, self.tourists[touristId].endSpot):
                reward=-20
            return False, reward, commuteTime
        else:
            # 如果能去这个景点
            if self.checkAvailabilityAct(touristId, action):
                commuteTime=self.calculateDistance(touristId, self.spots[action])/self.tourists[touristId].speed
                r=self.spots[action].rMax
                reward=r
                # 如果去了这个景点还能去终点
                if self.checkCompleteAvailability(touristId, action):
                    # 如果还未开始游览
                    if len(self.tourists[touristId].route) == 0:
                        if action == self.tourists[touristId].startSpot:
                            reward = 2*reward if reward>=0 else reward
                        else:
                            reward = -2*reward if reward>=0 else 2*reward
                    # 如果已经开始游览
                    else:
                        # 如果去过这个景点了
                        if action in self.tourists[touristId].route:
                            reward = -2*reward if reward>=0 else 2*reward
                        # 还没去过
                        else:
                            # 是终点
                            if action == self.tourists[touristId].endSpot:
                                reward = 2*reward if reward>=0 else -2*reward
                            # 不是终点
                            else:
                                # 是其他游客的起点或终点
                                if action != self.tourists[touristId].startSpot and action != self.tourists[touristId].endSpot and action in self.vipSpots:
                                    reward = 0.8*reward if reward>=0 else reward
                                # 不是其他游客的起点或终点
                                else:
                                    reward = reward if reward>=0 else reward
                    return True, reward, commuteTime
                # 如果去了这个景点就不能去终点了
                else:
                    if action == self.tourists[touristId].endSpot:
                        if len(self.tourists[touristId].route)!=0: 
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

        touristId, action =action[0], action[1]
        r_overtime=[]
        reward=0.0
        done = False
        nextTourist=-1        

        # ====================================================================
        # 执行动作
        # ====================================================================
        avail, reward, commuteTime =self.calculateReward_dynamic(touristId, action)
        if avail:
            # 如果能执行，就执行动作
            result=self.tourists[touristId].act(env=self, action=action, reward=reward, commuteTime=commuteTime)
            if result:
                sReward=self.calculateStaticReward(touristId, action)
                self.tourists[touristId].staticReward.append(sReward)
            # 如果确定这个动作能执行，从上次游览的景点中退出来
            if result and len(self.tourists[touristId].route)>1:
                self.spots[self.tourists[touristId].route[-2]].num-=1 
        else:
            # 游客不能一直等待，不往下执行其他游客也出不来，相当于死锁了
            self.tourists[touristId].timer.useTime(0.1)
        # ====================================================================
        # 执行完动作后检查时间，更新完成状态
        # ====================================================================
        total_not_avail = True
        if self.tourists[touristId].endSpot in self.tourists[touristId].route:
            self.tourists[touristId].complete=1
            if len(self.tourists[touristId].route)>0:
                self.spots[self.tourists[touristId].route[-1]].num-=1            
        else:
            if self.tourists[touristId].timer.getElapsed()>0:
                # for i in range(len(self.spots)):
                #     avail = not self.checkCompleteAvailability(touristId, i)
                #     total_not_avail = total_not_avail and avail
                # if total_not_avail:
                #     self.tourists[touristId].complete=1
                #     if len(self.tourists[touristId].route)>0:
                #         self.spots[self.tourists[touristId].route[-1]].num-=1            
                #         done=True
                pass
            else:
                self.tourists[touristId].complete=1
                if len(self.tourists[touristId].route)>0:
                    self.spots[self.tourists[touristId].route[-1]].num-=1            
                done=True
        # ====================================================================
        # 选择下一个执行动作的tourist
        # ====================================================================
        self.tourists.sort(key=lambda tourist: tourist.timer.nextFinish)
        for tourist in self.tourists:
            if tourist.complete==0 and tourist.timer.checkAvailability():
                nextTourist=tourist.touristId
                break
        # 选出来下一个执行动作的 tourist 之后，把 tourist list 重新按 touristId 排序，才能通过下标选出来正确的 tourist
        self.tourists.sort(key=lambda tourist: tourist.touristId)

        # r_overtime 的值应该与之后给的 reward 相匹配
        # ====================================================================
        # 生成观察
        # ====================================================================
        if nextTourist != -1:
            total_not_avail=True
            for spot in self.spots:
                able=self.checkCompleteAvailability(nextTourist, spot.spotId)
                # if spot.spotId != self.tourists[nextTourist].endSpot:
                #     total_not_avail = total_not_avail and (() and (not able))
                # 如果去了这个景点之后还能去终点
                if able:
                    avail, reward, commuteTime =self.calculateReward_static(nextTourist, spot.spotId)
                    rpt = reward/(commuteTime+spot.timeCost)
                else:
                    avail, reward, commuteTime =self.calculateReward_static(nextTourist, spot.spotId)
                    rpt = reward
                r_overtime.append(rpt)
                total_not_avail = total_not_avail and (not (reward>0 and able and avail))
            # 如果去完任何一个景点之后都不能再去终点了，那么终点的reward变大
            if total_not_avail and len(self.tourists[nextTourist].route)>0:
            # if self.tourists[nextTourist].timer.getElapsed()<2*self.spots[self.tourists[nextTourist].endSpot].timeCost:
                r_o=[-2*rpt if rpt>0 else rpt for rpt in r_overtime]
                r_o[self.tourists[nextTourist].endSpot]=2*spot.rMax
                r_overtime=r_o
        else:
            for i in range(len(self.spots)):
                r_overtime.append(0)
        r_overtime.append(0)
        return r_overtime, reward, done, nextTourist
    
    def checkAvailabilityAct(self, touristId, action):
        if self.tourists[touristId].timer.getElapsed()<0:
            # 如果用户所剩总时间为负，肯定不够执行动作action了
            return False
        if action==len(self.spots)+1:
            # 动作是等待的话，时间消耗可视为0，因为最低可以等待0秒
            return True
        else:
            timeCost=self.spots[action].timeCost
        if len(self.tourists[touristId].route)==0:
            pos=self.tourists[touristId].start_point
        else:
            pos=self.spots[self.tourists[touristId].route[-1]].location
        remain=self.tourists[touristId].timer.getElapsed()-timeCost
        
        if remain>=math.hypot(pos[0] - self.spots[action].location[0], pos[1] - self.spots[action].location[1])/self.tourists[touristId].speed:
            return True
        else:
            return False

    def checkCompleteAvailability(self, touristId, spotId):
        if spotId != self.tourists[touristId].endSpot:
            if len(self.tourists[touristId].route)>0:
                pos=self.spots[self.tourists[touristId].route[-1]].location
            else:
                pos=self.tourists[touristId].start_point
            et=self.tourists[touristId].timer.getElapsed()-self.spots[self.tourists[touristId].endSpot].timeCost - math.hypot(pos[0] - self.spots[spotId].location[0], pos[1] - self.spots[spotId].location[1])/self.tourists[touristId].speed
            pos=self.spots[spotId].location
            et=et-self.spots[spotId].timeCost - math.hypot(pos[0] - self.spots[self.tourists[touristId].endSpot].location[0], pos[1] - self.spots[self.tourists[touristId].endSpot].location[1])/self.tourists[touristId].speed
            return et>=0
        else:
            # 选择返回false的原因是主要检查除了终止节点以外的节点
            return False

    def calculateDistance(self, touristId, s: Spot):
        if len(self.tourists[touristId].route)==0:
            pos=self.tourists[touristId].start_point
        else:
            pos=self.spots[self.tourists[touristId].route[-1]].location
        return math.hypot(pos[0] - s.location[0], pos[1] - s.location[1])

    def reset(self):
        # spots, tourists=generateData()
        # self.spots = spots
        # self.tourists = tourists
        spots = self.spots
        tourists = self.tourists
        r_overtime=[]
        self.tourists.sort(key=lambda tourist: tourist.timer.nextFinish)
        touristId=self.tourists[0].touristId
        for s in spots:
            commuteTime=(math.hypot(self.tourists[touristId].start_point[0] - s.location[0], self.tourists[touristId].start_point[1] - s.location[1]))/self.tourists[touristId].speed
            r=s.reward()
            if s.spotId==self.tourists[touristId].startSpot:
                reward = 5*r if r>=0 else r
                r_overtime.append(reward)
            else:
                reward = -2*r if r>=0 else 2*r
                r_overtime.append(reward)
        # 什么也不做的 reward
        r_overtime.append(0)
        self.tourists.sort(key=lambda tourist: tourist.touristId)
        return r_overtime, touristId
        # return None, None


    def render(self):
        # Implement viz
        pass