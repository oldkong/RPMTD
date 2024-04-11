photo_num=[107, 53, 116, 12, 86, 135, 898, 136, 59, 517, 61, 51, 104, 216, 122, 1327, 8, 169, 306, 694, 1858, 300, 479, 36, 97, 216, 493, 88, 29, 2057, 3750, 787, 170, 76, 22, 492, 11, 33, 973, 67, 776, 95, 3, 220, 1005, 431, 201, 252, 752, 38, 1377, 472, 81, 5, 3392, 226, 525, 281, 766, 477, 278, 100, 2137, 903, 531, 278, 26, 249, 380, 415, 4891, 47]
tourist_num=[13, 5, 8, 5, 11, 15, 119, 13, 7, 83, 14, 10, 74, 22, 11, 66, 1, 8, 37, 79, 221, 16, 50, 13, 18, 20, 24, 7, 6, 132, 150, 41, 18, 14, 5, 22, 3, 3, 110, 15, 86, 21, 1, 37, 119, 37, 23, 35, 78, 68, 134, 50, 14, 4, 242, 52, 79, 35, 177, 52, 18, 9, 168, 45, 46, 49, 8, 64, 14, 32, 325, 3]

# photo per tourist
assert len(photo_num)==len(tourist_num)
ppt=[photo_num[i]/tourist_num[i] for i in range(len(photo_num))]


# *********************************************************************************************************************************


poi_aoi_path="/home/kong/work/0_marrrl/multi_v7/data/POI_AOI.csv"
piview_path="/home/kong/work/0_marrrl/multi_v7/data/1H_PIview.csv"


poi_aoi_file=open(poi_aoi_path)
piview_file=open(piview_path)


poi_aoi={}
for l in poi_aoi_file.readlines()[1:]:
    items=l.strip().split(",")
    poi_aoi[items[0]]=int(items[1])

piview=[]
for i in range(24):
    pl=[]
    pl.append([0 for i in range(72)])
    pl.append([0.0 for i in range(72)])
    piview.append(pl)
# piview[0]-piview[23] 是不同时刻的景点人数和评分数据
# piview[0][0] 是0点各个景点的人数数组，piview[0][1]是0点各个景点的评分数组
# piview[0][0][i] 是0点时景点i的人数，piview[0][1][i]是0点各个景点的评分
for l in piview_file.readlines()[1:]:
    try:
        # pid,timestamp,score_mean,photo_count
        items=l.strip().split(",")
        t=int(items[1].split(":")[0])
        aoi=poi_aoi[items[0]]
        score_mean=float(items[2])
        count=int(items[3])
        piview[t][0][aoi]+=count
        piview[t][1][aoi]+=score_mean*count
    except:
        print(l)

for t in range(24):
    for aoi in range(72):
        if piview[t][0][aoi] != 0:
            piview[t][1][aoi]=piview[t][1][aoi]/piview[t][0][aoi]
        else:
            print(piview[t][1][aoi], piview[t][0][aoi])
        print("ppt: ", ppt)
        piview[t][0][aoi]=int(piview[t][0][aoi]/ppt[aoi])

outfile=open("statistic_v3.txt", "w")
not_null_spot=[]
for t in range(24):
    c=0
    for item in piview[t][0]:
        if item!=0:
            c+=1
    not_null_spot.append(c)

import numpy as np
thres=np.mean(not_null_spot)
indicator=[nns>1.5*thres for nns in not_null_spot]
# outfile.write(not_null_spot.__str__()+"\n")


for t in range(24):
    if indicator[t]:
        piview[t][0]=[ p+int(0.1*thres) for p in piview[t][0] ]
        # outfile.write(str(t)+", "+str(len(piview[t][0]))+", "+"\n")
        outfile.write(str(t)+" \tnum\t")
        outfile.write(piview[t][0].__str__())
        outfile.write("\n")
        npm=np.mean(piview[t][1])
        for i in range(len(piview[t][1])):
            print(piview[t][1][i])
            # if piview[t][1][i]==0.0:
            piview[t][1][i]+=npm
        outfile.write(str(t)+"\tscore\t")
        outfile.write(piview[t][1].__str__())
        outfile.write("\n")
        

sum_count=[0 for i in range(72)]
sum_score=[0.0 for i in range(72)]
for t in range(24):
    for aoi in range(72):
        sum_count[aoi]+=piview[t][0][aoi]
        sum_score[aoi]+=piview[t][1][aoi]

# outfile.write(sum_count.__str__())
# outfile.write("\n")

# outfile.write(sum_score.__str__())
# outfile.write("\n")


