# -*- coding: utf-8 -*-
from dqn import DQN, parse_args

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.animation as animation
import argparse
from datetime import datetime
import os
from drl_base_v7 import gini


# info="MARLRR,\nIdentical,\n$|\mathcal{T}|$=100"
info="MARLRR,\nSimilar,\n$|\mathcal{T}|$=200"

def edit_distance(la, lb):
    if la == lb:
        return 0
    if len(la) == 0:
        return len(lb)
    if len(lb) == 0:
        return len(la)
    # 初始化dp矩阵
    dp = [[0 for _ in range(len(la) + 1)] for _ in range(len(lb) + 1)]
    # 当a为空，距离和b的长度相同
    for i in range(len(lb) + 1):
        dp[i][0] = i
    # 当b为空，距离和a和长度相同
    for j in range(len(la) + 1):
        dp[0][j] = j
    # 递归计算
    for i in range(1, len(lb) + 1):
        for j in range(1, len(la) + 1):
            dp[i][j] = dp[i-1][j-1]
            if la[j-1] != lb[i-1]:
                dp[i][j] = min([dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]) + 1
    return dp[len(lb)][len(la)]




def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default="dqn_200_2022.03.02_17_30.h5")
    # parser.add_argument('--model_name', default="dqn_200_2022.06.01_14_47.h5")
    parser.add_argument('--model_name', default="dqn_0_2022.06.06_22_18.h5")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    now = datetime.now()
    args=parse_args()
    model_name=args.model_name
    # timestamp=now.strftime('%Y.%m.%d_%H_%M')
    # dir="visual"
    timestamp=now.strftime('%Y.%m.%d_%H_%M_%S')
    # dir="visual/"+timestamp+"_"+args.model_name
    dir="visual/"+"_"+args.model_name
    if not os.path.exists(dir):
        os.mkdir(dir)

    model = DQN()
    model.load(filename='model/'+model_name)
    r_sum, s_r_sum, g, s_g = model.play()

    env=model.env
    spots=env.spots
    tourists=env.tourists
    
    log=env.log
    logname = dir+"/dynamicReward"+".log"
    os.rename(log.name, logname)
    log.write("*"*100)
    for s in spots:
        log.write(s.__str__()+"\n")
    log.write("*"*100+"\n"+"*"*100+"\n"+"*"*100+"\n")
    for t in tourists:
        log.write(t.__str__()+"\n")
    log.write("*"*100+"\n")
    log.write("Total reward: "+str(r_sum)+"\n")
    log.write("Gini: "+str(g)+"\n")
    log.write("Total static reward: "+str(s_r_sum)+"\n")
    log.write("Gini static: "+str(s_g)+"\n")

    colors=np.random.uniform(15, 80, len(spots))

# ------------------------------------------
    # ydynamic=[-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ydynamic=[-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    # yspot=[0, 25, 50, 75, 100, 125, 150, 175, 200]
    # yspot=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    # yspot=[0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325]
    yspot=[0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    ystatic=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    # yspot=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
# ------------------------------------------

    # 游客动态回报柱状图*起***********************************************************
    bar = plt.figure(figsize=(20, 12.36))

    x=[t.touristId for t in tourists]
    y=[sum(t.reward) for t in tourists]

    plt.bar(x, y, color='c', label='a')
    # plt.plot(x, y, 'r-', label='b')    
    plt.xlabel("Tourist", fontsize=30)
    plt.ylabel("Dynamic Reward", fontsize=30)
    plt.xticks(fontsize=20)
    plt.ylim([ydynamic[0],ydynamic[-1]])
    plt.yticks(ydynamic, [str(y) for y in ydynamic], fontsize=20)
    pp = PdfPages(dir+"/_tourist_dynamic_reward_bar.pdf")
    plt.margins(0.02,0.02)
    pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
    pp.close()
    # 游客动态回报柱状图*止***********************************************************


    # 游客静态回报柱状图*起***********************************************************
    bar = plt.figure(figsize=(20, 12.36))

    x=[t.touristId for t in tourists]
    y=[sum(t.staticReward) for t in tourists]

    plt.bar(x, y, color='c', label='a')
    # plt.plot(x, y, 'r-', label='b')
    plt.xlabel("Tourist", fontsize=30)
    plt.ylabel("Static Reward", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    pp = PdfPages(dir+"/_tourist_static_reward_bar.pdf")
    plt.margins(0.02,0.02)
    # plt.title("trained with Dynamic Reward", fontsize=40)
    plt.yticks(ystatic, [str(y) for y in ystatic], fontsize=20)
    pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
    pp.close()
    # 游客静态回报柱状图*止***********************************************************



    # 景点柱状图*起***********************************************************
    bar = plt.figure(figsize=(20, 12.36))

    x=[s.spotId for s in spots]
    y=[0 for i in range(len(spots))]
    c=[s.cMax for s in spots]

    for t in tourists:
        for r in t.route:
            y[r]+=1.0
    y_c=[y[i]/c[i] for i in range(len(spots))]
    y_c_gini=gini(y_c)
    plt.bar(x, y, color='c', label="$G_{s}$: "+str(round(y_c_gini, 2)))
    # plt.plot(x, y, 'r-', label='b')
    log.write("+"*100+"\n")
    log.write("$G_{s}$: "+str(y_c_gini)+"\n")
    plt.legend(fontsize=50)
    plt.xlabel("Spot", fontsize=40)
    plt.ylabel("Number of Visit", fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylim([yspot[0],yspot[-1]])
    plt.yticks(yspot, [str(y) for y in yspot], fontsize=40)
    # plt.text(0.03, 0.82, info, fontsize=40, transform = plt.gca().transAxes)
    
    pp = PdfPages(dir+"/_spot_bar.pdf")
    plt.margins(0.02,0.02)
    pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
    pp.close()
    # 景点柱状图*止***********************************************************


    # 地图*起***********************************************************
    ed=0
    for i in range(len(tourists)):
        for j in range(i, len(tourists)):
            ed+=edit_distance(tourists[i].realRoute, tourists[j].realRoute)
            print(ed)
    ed=ed/(len(tourists)+0.0)
    print(ed)
    log.write("+"*100+"\n")
    log.write("$edit distance of routes$: "+str(ed)+"\n")
    fig = plt.figure(figsize=(10, 10))
    x=[s.location[0] for s in spots]
    y=[s.location[1] for s in spots]
    # colors=np.random.uniform(15, 80, 91)
    # plt.scatter([x[0]], [y[0]], marker='o', color=["white"], label="Edit Distance of Routes: "+str(round(ed, 2)))
    plt.scatter(x, y, marker='o', c=colors)
    # plt.legend(fontsize=12)

    # plt.text(0.03, 0.06 , info, fontsize=40, transform = plt.gca().transAxes)                

    plt.text(0.03, 0.94 , "$ED_{r}$: "+str(round(ed, 2)), fontsize=30, transform = plt.gca().transAxes)                

    for s in spots:
        plt.annotate(s.spotId, (s.location[0], s.location[1]), fontsize=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks([])
    plt.yticks([])
    for t in tourists:
        tx=[spots[r].location[0] for r in t.route]
        ty=[spots[r].location[1] for r in t.route]
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
    pp = PdfPages(dir+"/_map.pdf")
    plt.margins(0.02,0.02)
    pp.savefig(fig, bbox_inches = 'tight', pad_inches = 0)
    pp.close()

    # 地图*止***********************************************************


    # # 动图*起***********************************************************

    # gif = plt.figure(figsize=(10, 10))
    # plt.scatter(x, y, marker='o', c=colors)
    # for s in spots:
    #     plt.annotate(s.spotId, (s.location[0], s.location[1]))

    # ims, buf = [], []
    # for t in tourists:
    #     tx=[spots[r].location[0] for r in t.route]
    #     ty=[spots[r].location[1] for r in t.route]
    #     x_length=[tx[i+1]-tx[i] for i in range(len(tx)-1)]
    #     y_length=[ty[i+1]-ty[i] for i in range(len(ty)-1)]
    #     try:
    #         del tx[-1]
    #         del ty[-1]
    #         im = plt.quiver(tx, ty, x_length, y_length, color="C"+str(t.touristId), angles='xy', scale_units='xy', scale=1, width=0.005)
    #     except:
    #         print("PROBLEM OCCUR:")
    #         print("TX::: ", tx)
    #         print("TY::: ", ty)
    #         print("x_length::: ", x_length)
    #         print("y_length::: ", y_length)
    #         print("T.ROUTE::: ", t.route)            
    #     buf.append(im)
    # for i in range(len(buf)):
    #     cache=[]
    #     for j in range(i):
    #         cache.append(buf[j])
    #     ims.append(cache)
    # ani = animation.ArtistAnimation(gif, ims, interval=200, blit=False)
    # ani.save(dir+"/_ani.gif", writer='pillow')
    # # 动图*止***********************************************************



