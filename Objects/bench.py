class Bench():
    def __init__(self,tonnage,node,x=0,y=0,z=0,grade = 0):
        self.x=x
        self.y=y
        self.z=z
        self.grade=grade
        self.resource = None
        self.current_tonnage = tonnage
        self.node = node
        