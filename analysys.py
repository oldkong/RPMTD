import os

process_list = []

result = {}
result['Gini']= []
result['Total static reward']= []
result['Gini static']= []
result['Total reward']= []

def path_read_log(path):
  # input：文件夹路径
  # output: print 这个路径下所有log文件里的最终数据
  l = ['Total ', 'Gini:', 'Total', 'Gini']

  file_list=os.listdir(path)

  for i in file_list:
    if os.path.splitext(i)[1]=='.log':
      # print(i)
      file_path = os.path.join(path,i)

      sp = open(file_path)
      for line in sp.readlines():
        if line.split(' ')[0] in l:
          # print(line)
          result[line.split(':')[0]].append(float(line.split(':')[1].split('\n')[0]))
      print('*'*20)
      

def read_log(folder_path, key_strings):

  folder=os.listdir(folder_path)
  key_strings = key_strings

  for item in folder:
    if os.path.splitext(item)[-1] != '.log': #判断是否是文件夹
      for ks in key_strings:
        if ks in item:
          path = os.path.join('/home/y.kong/_Congestion-aware_Route_Recommendation/visual',item) # 路径
          process_list.append(item)
          path_read_log(path)


def mean(text):
  l = result[text]
  return sum(l)/len(l)



folder_path = '/home/y.kong/_Congestion-aware_Route_Recommendation/visual'

key_strings = ['dqn_5_2022.06.05_15_23']  #######修改关键字

read_log(folder_path,key_strings)


for item in process_list:
  print(item)



Gini_mean = mean('Gini')
Total_static_reward_mean = mean('Total static reward')
Gini_static_mean = mean('Gini static')
Total_reward_mean = mean('Total reward')

print(len(process_list),'files are processed')  

print('Gini_mean:',Gini_mean)
print('Total_static_reward_mean:',Total_static_reward_mean)
print('Gini_static_mean:',Gini_static_mean)
print('Total_reward_mean:',Total_reward_mean)