# -*- coding: utf-8 -*-
from dqn import DQN

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.animation as animation
import argparse
from datetime import datetime
import os
 
from catrr import RREnv, generateData

info="MARLRR,\nRandom\n"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="dqn.h5")
    parser.add_argument('--epoch', default="400")
    parser.add_argument('--avgEopch', default="3")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    now = datetime.now()
    args=parse_args()
    model_name=args.model_name
    epoch=int(args.epoch)
    avgEopch=int(args.avgEopch)
    timestamp=now.strftime('%Y.%m.%d_%H_%M')
    dir="visual/"+timestamp
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    spotNum=72

    r_sum_mean_array=[]
    s_r_sum_mean_array=[]
    log = open(dir+"/_.log", "a")
    for i in range(1, epoch+1):
        r_sum_array=[]
        s_r_sum_array=[]
        visitNum=[0 for i in range(spotNum)]
        for j in range(avgEopch):
            spots, tourists=generateData(tNum=i)
            env=RREnv(spots, tourists)
            model = DQN(env)
            model.load(filename='model/'+model_name+'.h5')

            # print("Number of spots, tourists: ", len(model.env.spots),"  ",  len(model.env.tourists))
            r_sum, s_r_sum, g, s_g = model.play()
            r_sum_array.append(r_sum)
            s_r_sum_array.append(s_r_sum)
            for t in model.env.tourists:
                for r in t.route:
                    visitNum[r]+=1
        r_sum_mean=np.mean(r_sum_array)
        s_r_sum_mean=np.mean(s_r_sum_array)
        r_sum_mean_array.append(r_sum_mean)
        s_r_sum_mean_array.append(s_r_sum_mean)
        print(i, " ::: ", r_sum_mean, " ::: ", s_r_sum_mean)
        log.write(str(i)+"\t"+str(r_sum_mean)+"\t"+str(s_r_sum_mean)+"\n")


        visitNum=np.divide(visitNum, float(avgEopch))
        log.write("Visit Times in Epoch "+str(i)+"\n")
        log.write(visitNum.__str__()+"\n")
        if i%100==0:
            bar = plt.figure(figsize=(20, 12.36))
            plt.bar(range(1, spotNum+1), visitNum, color='c', label='Number of Visit')
            plt.legend(fontsize=60)
            plt.grid()
            plt.xlabel("Spot", fontsize=60)
            # plt.ylabel("Number of Visits", fontsize=30)
            plt.xticks(fontsize=60)
            plt.yticks(fontsize=60)
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)

            pp = PdfPages(dir+"/dis_"+str(i)+".pdf")
            plt.margins(0.02,0.02)
            pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
            pp.close()
            plt.close()


        if i%100==0:
            log.write("r_sum_mean_array"+"\n")
            log.write(str(r_sum_mean_array))
            log.write("s_r_sum_mean_array"+"\n")
            log.write(str(s_r_sum_mean_array))

            index=range(1, i+1)
            trend=plt.figure(figsize=(20, 12.36))
            # plt.title("")
            plt.xlabel("Number of tourist", fontsize=60)
            plt.text(0.03, 0.4 , info, fontsize=60, transform = plt.gca().transAxes)                
            plt.xticks(fontsize=60)
            plt.yticks(fontsize=60)
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.ylabel("Total Reward", fontsize=60)
            plt.plot(index, r_sum_mean_array, '-', color='red', label="$\mathcal{R}$")
            plt.plot(index, s_r_sum_mean_array,'-', color='green', label="$\mathcal{R}_{s}$")
            plt.legend(fontsize=60)
            plt.grid()
            pp = PdfPages(dir+"/"+timestamp+"_trend_"+str(i)+".pdf")
            plt.margins(0.02,0.02)
            pp.savefig(trend, bbox_inches = 'tight', pad_inches = 0)
            pp.close()
            plt.close()



    log.write("r_sum_mean_array"+"\n")
    log.write(str(r_sum_mean_array))
    log.write("s_r_sum_mean_array"+"\n")
    log.write(str(s_r_sum_mean_array))

    index=range(1, epoch+1)
    trend=plt.figure(figsize=(20, 12.36))
    # plt.title("")
    plt.xlabel("Number of tourist")
    plt.xticks(rotation=45)
    plt.ylabel("Total Reward")
    plt.plot(index, r_sum_mean_array, '-', color='red', label="reward")
    plt.plot(index, s_r_sum_mean_array,'-', color='green', label="static reward")
    plt.legend()
    plt.grid()
    pp = PdfPages(dir+"/"+timestamp+"_trend.pdf")
    plt.margins(0.02,0.02)
    pp.savefig(trend, bbox_inches = 'tight', pad_inches = 0)
    pp.close()
    plt.close()

















    # # 游客动态回报柱状图*起***********************************************************
    # bar = plt.figure(figsize=(10, 10))

    # x=[t.touristId for t in tourists]
    # y=[sum(t.reward) for t in tourists]

    # plt.bar(x, y, color='c', label='a')
    # plt.plot(x, y, 'r-', label='b')    
    # plt.xlabel("Tourist", fontsize=30)
    # plt.ylabel("Dynamic Reward", fontsize=30)
    # plt.xticks(fontsize=20)
    # plt.ylim([ydynamic[0],ydynamic[-1]])
    # plt.yticks(ydynamic, [str(y) for y in ydynamic], fontsize=20)
    # pp = PdfPages("visual/dynamicReward_"+timestamp+"_tourist_dynamic_reward_bar.pdf")
    # plt.margins(0.02,0.02)
    # pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
    # pp.close()
    # # 游客动态回报柱状图*止***********************************************************


    # # 游客静态回报柱状图*起***********************************************************
    # bar = plt.figure(figsize=(10, 10))

    # x=[t.touristId for t in tourists]
    # y=[sum(t.staticReward) for t in tourists]

    # plt.bar(x, y, color='c', label='a')
    # plt.plot(x, y, 'r-', label='b')
    # plt.xlabel("Tourist", fontsize=30)
    # plt.ylabel("Static Reward", fontsize=30)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # pp = PdfPages("visual/dynamicReward_"+timestamp+"_tourist_static_reward_bar.pdf")
    # plt.margins(0.02,0.02)
    # # plt.title("trained with Dynamic Reward", fontsize=40)
    # pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
    # pp.close()
    # # 游客静态回报柱状图*止***********************************************************



    # # 景点柱状图*起***********************************************************
    # bar = plt.figure(figsize=(10, 10))

    # x=[s.spotId for s in spots]
    # y=[0 for i in range(len(spots))]

    # for t in tourists:
    #     for r in t.route:
    #         y[r]+=1

    # plt.bar(x, y, color='c', label='a')
    # plt.plot(x, y, 'r-', label='b')
    # plt.xlabel("Spot", fontsize=30)
    # plt.ylabel("Number of Visits", fontsize=30)
    # plt.xticks(fontsize=20)
    # plt.ylim([yspot[0],yspot[-1]])
    # plt.yticks(yspot, [str(y) for y in yspot], fontsize=20)

    # pp = PdfPages("visual/dynamicReward_"+timestamp+"_spot_bar.pdf")
    # plt.margins(0.02,0.02)
    # pp.savefig(bar, bbox_inches = 'tight', pad_inches = 0)
    # pp.close()
    # # 景点柱状图*止***********************************************************


    # # 地图*起***********************************************************

    # fig = plt.figure(figsize=(10, 10))

    # x=[s.location[0] for s in spots]
    # y=[s.location[1] for s in spots]
    # # colors=np.random.uniform(15, 80, 91)
    # plt.scatter(x, y, marker='o', c=colors)
    # for s in spots:
    #     plt.annotate(s.spotId, (s.location[0], s.location[1]), fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=20)
    # for t in tourists:
    #     tx=[spots[r].location[0] for r in t.route]
    #     ty=[spots[r].location[1] for r in t.route]
    #     x_length=[tx[i+1]-tx[i] for i in range(len(tx)-1)]
    #     y_length=[ty[i+1]-ty[i] for i in range(len(ty)-1)]
    #     try:
    #         del tx[-1]
    #         del ty[-1]
    #         plt.quiver(tx, ty, x_length, y_length, color="C"+str(t.touristId), angles='xy', scale_units='xy', scale=1, width=0.005)
    #     except:
    #         print("PROBLEM OCCUR:")
    #         print("TX::: ", tx)
    #         print("TY::: ", ty)
    #         print("x_length::: ", x_length)
    #         print("y_length::: ", y_length)
    #         print("T.ROUTE::: ", t.route)
    # pp = PdfPages("visual/dynamicReward_"+timestamp+"_map.pdf")
    # plt.margins(0.02,0.02)
    # pp.savefig(fig, bbox_inches = 'tight', pad_inches = 0)
    # pp.close()

    # # 地图*止***********************************************************


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
    # ani.save("visual/dynamicReward_"+timestamp+"_ani.gif", writer='pillow')
    # # 动图*止***********************************************************



