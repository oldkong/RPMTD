aoi_old_new_file=open("/home/kong/work/0_marrrl/multi_v7/data/aoi_old_new.txt", "r")
aoi_old_new={}
for l in aoi_old_new_file.readlines():
    ls=l.strip().split(",")
    aoi_old_new[ls[0]]=ls[1]

img_all_info_path="/home/kong/work/0_marrrl/multi_v7/data/Img_all_info.csv"

img_all_info=open(img_all_info_path)

img={}
for l in img_all_info.readlines()[1:]:
    items=l.strip().split(",")
    id=items[0]
    score=items[1]
    poi=items[2]
    aoi=aoi_old_new[items[3]]
    if aoi!="*" and score!="" and score !=" ":
        key= id+", "+aoi
        value=score
        img[key]=value



filtered_img=open("/home/kong/work/0_marrrl/multi_v7/data/filtered_img.txt", "w+")

for k, v in img.items():
    filtered_img.write(k+", "+v+"\n")

num=[0 for i in range(72)]
score=[0.0 for i in range(72)]

for k, v in img.items():
    aoi=int(k.split(", ")[1])
    s=float(v)
    num[aoi]+=1
    score[aoi]+=s

for i in range(72):
    if num[i]!=0:
        score[i]=score[i]/num[i]

import numpy as np

filtered_img.write(num.__str__()+"\n")
filtered_img.write("Mean num: "+str(np.mean(num))+"\n")

filtered_img.write(score.__str__()+"\n")
filtered_img.write("Mean score: "+str(np.mean(score))+"\n")

sn={}
for n, s in zip(num, score):
    # filtered_img.write(str(n)+", "+str(s)+"\n")
    sn[s]=n
sorted_key=sorted(sn)
for k in sorted_key:
    # filtered_img.write(str(s)+", "+str(n)+"\n")
    filtered_img.write(str(k)+", "+str(sn[k])+"\n")


