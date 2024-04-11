# -*- coding: utf-8 -*-
from dqn_static_reward import DQN

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.animation as animation
import argparse
from datetime import datetime
import os

from catrr_base import RREnv, generateData

info="baseline,\nRandom\n"

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
            # plt.xticks(rotation=45)
            plt.ylabel("Total Reward", fontsize=60)
            plt.xticks(fontsize=60)
            plt.yticks(fontsize=60)
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
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
    plt.plot(index, r_sum_mean_array, '-', color='red', label="$\mathcal{R}$")
    plt.plot(index, s_r_sum_mean_array,'-', color='green', label="\mathcal{R}_{s}")
    plt.legend()
    plt.grid()
    pp = PdfPages(dir+"/"+timestamp+"_trend.pdf")
    plt.margins(0.02,0.02)
    pp.savefig(trend, bbox_inches = 'tight', pad_inches = 0)
    pp.close()
    plt.close()
