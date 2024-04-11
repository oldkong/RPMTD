from timer_v6 import Timer
from rrEnv_v6 import RREnv, Spot, Tourist, generateDate
import math

env = RREnv()
spots, tourist=generateDate()
dists=[]
for s in spots:
    dist=[]
    for s2 in spots:
        dist.append(math.hypot(s.location[0] - s2.location[0], s.location[1] - s2.location[1]))
    dists.append(dist)

episodes = 100
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 

    while not done:
        #env.render()
        for tourist in env.tourists:
            if tourist.spotTimer.getElapsed()>0:
                pass
            else:
                dist=dists[tourist.route[-1]]
                flag=False
                for i in range(len(dist)):
                    if tourist.timer.getElapsed()>env.spots[i].timeCost+dist[i]/tourist.speed:
                        flag=True
                        break
                if not flag:
                    done=True
                else:
                    action = env.action_space.sample()
                    while action in tourist.route:
                        action = env.action_space.sample()
                    n_state, reward, done, info = env.step(tourist, action)
                    score=tourist.reward
                    print("tourist.route: ", tourist.route)
    print('Episode:{} Score:{}'.format(episode, score))