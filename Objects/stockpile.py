import simpy
class Stockpile():
    def __init__(self,env,tonnage,node=None,x=0,y=0,z=0,grade = 0):
        self.x=x
        self.y=y
        self.z=z
        self.env = env
        self.grade=grade
        self.resource = simpy.Resource(self.env,capacity=1)
        self.current_tonnage = tonnage
        self.node = node
        