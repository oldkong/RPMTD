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

y=[0, 20000, 40000, 60000, 80000, 100000, 120000, 140000]
ylabel=["0", "20k", "40k", "60k", "80k", "100k", "120k", "140k"]
x=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
xlabel=["0", "0.25k", "0.5k", "0.75k", "1k", "1.25k", "1.5k", "1.75k", "2k"]

info="MARLRR,\nRandom\n"

spotNum=72

r=[]
rs=[]

for l in open("/home/kong/work/0_github/Congestion-aware_Route_Recommendation/visual/2022.03.26_14_35/marlrr").readlines():
    i=l.split(":::")
    r.append(float(i[1]))
    rs.append(float(i[2]))
    if int(i[0])==2000:
        break

index = range(1, 2001)

# index=range(1, i+1)
trend=plt.figure(figsize=(20, 12.36))
# plt.title("")
plt.xlabel("Number of tourist", fontsize=60)
# plt.text(0.03, 0.4 , info, fontsize=60, transform = plt.gca().transAxes)
plt.ylim((0, 140000))
plt.yticks(y, ylabel, fontsize=60)
plt.xticks(x, xlabel, fontsize=60)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.ylabel("Total Reward", fontsize=60)
plt.plot(index, r, '-', color='red', label="$\mathcal{R}$")
plt.plot(index, rs,'-', color='green', label="$\mathcal{R}_{s}$")
plt.legend(fontsize=60, loc=2)
plt.grid()
pp = PdfPages("_draw_trend_narlrr_"+str(2000)+".pdf")
plt.margins(0.02,0.02)
pp.savefig(trend, bbox_inches = 'tight', pad_inches = 0)
pp.close()
plt.close()

# ------------------------------------------------------------------------------------------

info="baseline,\nRandom\n"

spotNum=72

r=[]
rs=[]

for l in open("/home/kong/work/0_github/Congestion-aware_Route_Recommendation/visual/2022.03.26_14_35/baseline").readlines():
    i=l.split(":::")
    r.append(float(i[1]))
    rs.append(float(i[2]))
    if int(i[0])==2000:
        break

index = range(1, 2001)

# index=range(1, i+1)
trend=plt.figure(figsize=(20, 12.36))
# plt.title("")
plt.xlabel("Number of tourist", fontsize=60)
# plt.text(0.03, 0.4 , info, fontsize=60, transform = plt.gca().transAxes)
plt.ylim((0, 140000))
plt.yticks(y, ylabel, fontsize=60)
plt.xticks(x, xlabel, fontsize=60)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.ylabel("Total Reward", fontsize=60)
plt.plot(index, r, '-', color='red', label="$\mathcal{R}$")
plt.plot(index, rs,'-', color='green', label="$\mathcal{R}_{s}$")
plt.legend(fontsize=60, loc=2)
plt.grid()
pp = PdfPages("_draw_trend_baseline_"+str(2000)+".pdf")
plt.margins(0.02,0.02)
pp.savefig(trend, bbox_inches = 'tight', pad_inches = 0)
pp.close()
plt.close()

