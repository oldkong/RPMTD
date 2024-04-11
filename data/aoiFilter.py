aoi_data_file=open("/home/kong/work/0_marrrl/multi_v7/data/AOI_data.csv", "r+")

aoi_old_new_file=open("/home/kong/work/0_marrrl/multi_v7/data/aoi_old_new.txt", "r")
aoi_old_new={}
for l in aoi_old_new_file.readlines():
    ls=l.strip().split(",")
    old_aoi=ls[0]
    new_aoi=ls[1]
    aoi_old_new[ls[0]]=ls[1]
new=[]
for l in aoi_data_file.readlines():
    ls=l.strip().split(",")
    if aoi_old_new[ls[0]]!="*":
        ls[0]=aoi_old_new[ls[0]]
        l=ls[0]+", "+ls[1]+", "+ls[2]+"\n"
        new.append(l)
for n in new:
    aoi_data_file.write(n)