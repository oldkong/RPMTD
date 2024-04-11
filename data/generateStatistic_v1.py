
poi_aoi_path="/home/kong/work/0_marrrl/multi_v7/data/AOI_POI_all.csv"
# aoi_data_path="/home/kong/work/0_marrrl/multi_v7/data/AOI_data.csv"
plview_path="/home/kong/work/0_marrrl/multi_v7/data/1H_PIview.csv"


poi_aoi_file=open(poi_aoi_path)
# aoi_data_file=open(aoi_data_path)
plview_file=open(plview_path)


poi_aoi={}
for l in poi_aoi_file.readlines()[1:]:
    items=l.strip().split(",")
    poi_aoi[items[0]]=int(items[1])

plview=[]
for i in range(24):
    pl=[]
    pl.append([0 for i in range(90)])
    pl.append([0.0 for i in range(90)])
    plview.append(pl)
# plview[0]-plview[23] 是不同时刻的景点人数和评分数据
# plview[0][0] 是0点各个景点的人数数组，plview[0][1]是0点各个景点的评分数组
# plview[0][0][i] 是0点时景点i的人数，plview[0][1][i]是0点各个景点的评分
for l in plview_file.readlines()[1:]:
    try:
        # pid,timestamp,score_mean,photo_count
        items=l.strip().split(",")
        t=int(items[1].split(":")[0])
        aoi=poi_aoi[items[0]]
        score_mean=float(items[2])
        count=int(items[3])
        plview[t][0][aoi]+=count
        plview[t][1][aoi]+=score_mean*count
    except:
        print(l)

for t in range(24):
    for aoi in range(90):
        if plview[t][0][aoi] != 0:
            plview[t][1][aoi]=plview[t][1][aoi]/plview[t][0][aoi]
        else:
            print(plview[t][1][aoi], plview[t][0][aoi])

outfile=open("statistic.txt", "w")
null_spot=[]
for t in range(24):
    c=0
    for item in plview[t][0]:
        if item==0:
            c+=1
    null_spot.append(c)


outfile.write(null_spot.__str__()+"\n")


for t in range(24):
    outfile.write(str(t)+", "+str(len(plview[t][0]))+", "+"\n")
    outfile.write(plview[t][0].__str__())
    outfile.write("\n")
    outfile.write(plview[t][1].__str__())

    outfile.write("\n")

sum_count=[0 for i in range(90)]
sum_score=[0.0 for i in range(90)]
for t in range(24):
    for aoi in range(90):
        sum_count[aoi]+=plview[t][0][aoi]
        sum_score[aoi]+=plview[t][1][aoi]
outfile.write(sum_count.__str__())
outfile.write("\n")
outfile.write(sum_score.__str__())
outfile.write("\n")

aoi_data_file=open("/home/kong/work/0_marrrl/multi_v7/data/AOI_data.csv", "r+")
aoi_data=[]
i=0
aoi_old_new={}
for l in aoi_data_file.readlines():
    ls=l.strip().split(",")
    if sum_count[int(ls[0])]!=0:
        aoi_data.append(str(i)+", "+ls[1]+", "+ls[2])
        aoi_old_new[int(ls[0])]=i
        i+=1
    else:
        aoi_old_new[int(ls[0])]="*"
for aoi in aoi_data:
    aoi_data_file.write(aoi+"\n")

new_poi_aoi={}
for poi, aoi in poi_aoi.items():
    if aoi!="*":
        new_poi_aoi[poi]=aoi_old_new[aoi]

new_poi_aoi_file=open("/home/kong/work/0_marrrl/multi_v7/data/POI_AOI.csv", "w")
for poi, aoi in new_poi_aoi.items():
    new_poi_aoi_file.write(str(poi)+", "+str(aoi)+"\n")
