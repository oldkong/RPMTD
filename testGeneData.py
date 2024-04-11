spotStatistic=[[[], []] for i in range(7)]
spotStatisticFile=open("/home/kong/work/0_marrrl/multi_v7/data/statistic_v3.txt", "r")
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

print(spotStatistic)