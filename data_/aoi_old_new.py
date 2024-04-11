
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




sum_count=[0 for i in range(90)]
sum_score=[0.0 for i in range(90)]
for t in range(24):
    for aoi in range(90):
        sum_count[aoi]+=plview[t][0][aoi]
        sum_score[aoi]+=plview[t][1][aoi]


aoi_data_file=open("/home/kong/work/0_marrrl/multi_v7/data/AOI_data.csv", "r")
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

aoi_old_new_file=open("/home/kong/work/0_marrrl/multi_v7/data/aoi_old_new.txt", "w")

for k, v in aoi_old_new.items():
    aoi_old_new_file.write(str(k)+", "+str(v)+"\n")

