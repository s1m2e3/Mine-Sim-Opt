import simpy
import networkx as nx
import numpy as np

class Vehicle:

    def __init__(self,graph,env, x=0, y=0, z=0,node = None):
        self.x = x
        self.y = y
        self.z = z
        self.graph = graph
        self.status = 'idle'
        self.node = node
        self.env = env
        self.available_tasks = []
        
    def assign(self, task):
        #task includes the tuple origin node to task 
        #Make sure that tasks will only contain  valid destiny points
        self.task = task
        self.node = self.task[0]
        
    def get_travel_time(self):
        path  = nx.shortest_path(self.graph.graph,source=self.task[0],target=self.task[1])
        path = [(path[i],path[i+1]) for i in range(len(path)-1)]
        times = nx.get_edge_attributes(self.graph.graph,"times")
        travel_time=0
        for pair in path:
            travel_time += times[pair]
        return travel_time


class Truck(Vehicle):
    def __init__(self, x, y, z, capacity):
        super().__init__(x, y, z)
        self.capacity = capacity
        self.travel = self.env.process(self.traveling())
        self.type = "Truck"

    def get_time_to_discharge(self):
        return np.random.expo(3)

    
    def traveling(self):
        while True:
            if self.task is not None:

                travel_time = self.get_travel_time()
                self.status = "traveling"
                while travel_time:
                    yield self.env.timeout(travel_time)
                    travel_time = 0
                    self.task = self.task[1]
                    self.node = self.task
                    final_node = self.graph[self.task]["labels"]
                    index = final_node.find("_")
                    index = final_node[index:]
                    if "plant" in final_node:
                        
                        with self.graph.plants[index].resource.request():
                            discharge_time=self.get_time_to_discarge()
                            while discharge_time:
                                self.status="discharging"
                                yield self.env.timeout(discharge_time)
                                discharge_time = 0
                                self.status = "idle"
                                self.graph.plants[index].processed_tonnage += self.capacity
                                self.task = None


                    elif "stockpile" in final_node:
                        
                        with self.graph.stockpiles[index].resource.request():
                            discharge_time=self.get_time_to_discarge()
                            while discharge_time:
                                self.status = "discharging"
                                yield self.env.timeout(discharge_time)
                                discharge_time = 0
                                self.task = None
                                self.status = "idle"
                                self.graph.stockpiles[index].current_tonnage += self.capacity


                    elif "benches" in final_node:
                        
                        if self.graph.benches[index] is not None:
                            with self.graph.benches[index].resource.request():
                                load_time=self.get_time_to_load()
                                while load_time:
                                    self.status="loading"
                                    yield self.env.timeout(load_time)
                                    load_time = 0
                                    self.status = "idle"
                                    self.graph.benches[index].current_tonnage -= self.capacity
                                    self.task = None
                       
class Shovel(Vehicle):
    def __init__(self, x, y, z, production_rate):
        super().__init__(x, y, z)
        self.production_rate = production_rate
        
        self.resource = simpy.Resource(self.env,capacity=1)
        self.travel = self.env.process(self.traveling())
        self.type="Shovel"
    def get_shovel_travel_time(self):
        return self.get_travel_time()*3

    def assign_shovel_to_bench(self):
        
        #find bench in graph object given node of bench in graph.graph:
        if self.task != None:
            #find index of destination bench in graph.benches  given task 
            bench_index = [i for i in range(len(self.graph.bench)) if self.graph.bench.node == self.task[1]][0]
            
            #check if shovel in destiny bench:
            if self.node != self.graph.benches[bench_index].node:
                #travel to destination bench
                self.travel()
                self.graph.benches[bench_index].resource = self.resource
                if self.graph.benches[bench_index].current_tonnage > 0:
                    self.status="working"
                else:
                    self.status="idle"
                    self.task = None
        else:
            raise ValueError("need to give task to shovel before assigning shovel to bench")
        
    def travel(self):
        travel_time = self.get_shovel_travel_time()
        while True:
            if self.task is not None:
                self.status = "traveling"
                while travel_time:
                    yield self.env.timeout(travel_time)
                    travel_time = 0
                    self.task = self.task[1]
                    self.node = self.task
                break
         
              
