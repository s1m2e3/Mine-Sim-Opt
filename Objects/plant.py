import simpy

class Plant():
    def __init__(self,env,x=0,y=0,z=0):
        self.x=x
        self.y=y
        self.z=z
        self.env = env
        self.resource = simpy.Resource(self.env,capacity=1)
        self.processed_tonnage = 0
        self.node = None
        