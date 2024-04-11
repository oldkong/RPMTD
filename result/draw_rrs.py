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

y=[0, 20000, 40000, 60000, 80000, 120000, 140000]
ylabel=["0", "20", "40", "60", "80", "120", "140"]

info="MARLRR,\nRandom\n"

spotNum=72

r=[]
rs=[]

for l in open("RRS").readlines:
    i=l.split(":::")
    r.append(float(i[1]))
    rs.append(float(i[2]))
    if int(i)==2001:
        break

index = range(1, 2001)

index=range(1, i+1)
trend=plt.figure(figsize=(20, 12.36))
# plt.title("")
plt.xlabel("Number of tourist (/thousand)", fontsize=60)
plt.text(0.03, 0.4 , info, fontsize=60, transform = plt.gca().transAxes)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.ylabel("Total Reward", fontsize=60)
plt.plot(index, r, '-', color='red', label="$\mathcal{R}$")
plt.plot(index, rs,'-', color='green', label="$\mathcal{R}_{s}$")
plt.legend(fontsize=60)
plt.grid()
pp = PdfPages("_draw_trend_"+str(i)+".pdf")
plt.margins(0.02,0.02)
pp.savefig(trend, bbox_inches = 'tight', pad_inches = 0)
pp.close()
plt.close()

