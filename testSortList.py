import random

class People():
    def __init__(self, age=0):
        self.age=age
    
    def __str__(self) -> str:
        return "age: "+str(self.age)

ps=[]
for i in range(10):
    ps.append(People(age=random.random()))

for p in ps:
    print(p)

ps.sort(key=lambda p: p.age)
print("-"*100)
for p in ps:
    print(p)