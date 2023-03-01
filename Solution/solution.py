import networkx as nx
import random 
class Solution():
    def __init__(self,vehicles):
        #vehicles is a dictionary that contains a list of Vehicle objects

        self.vehicles = vehicles
        self.solution = {}
        

    def random_solution(self):
        
        self.update_available_tasks()
        available = self.get_available_tasks()
        
        for vehicle_type in available:
            already_used = []
            for vehicle in available[vehicle_type]:
                choice = random.choice(available[vehicle_type][vehicle], k=1)
                while choice in already_used:
                    choice = random.choice(available[vehicle_type][vehicle], k=1)
                already_used.append(choice)
                self.solution[vehicle]=choice
        return self.solution    

    def update_available_tasks(self,graph):
        
        labels = nx.get_node_attributes(graph,"labels")
        load_nodes = [i for i in labels if "benches" in labels[i]]
        discharge_nodes = [i for i in labels if "plant" or "stockpile" in labels[i]]
       

        for vehicle_type in self.vehicles:
            vehicle_node = [vehicle.node  if vehicle.task==None else vehicle.task[-1] for vehicle in self.vehicles[vehicle_type]] 

            #get type of current node
            vehicle_node_labels = [labels[node] for node in vehicle_node]
        
            for i in range(len(vehicle_node_labels)):
                #update available tasks in vehicle class
                #update solution dictionary
                if "plant" or "stockpile" in vehicle_node_labels[i]:
                    #current task ends in a discharge node, direct to load nodes
                    
                    self.vehicles[vehicle_type][i].available_tasks = [(i,j) for j in load_nodes]

                elif "benches" in vehicle_node_labels[i]:
                    #current task ends in load node, direct to discharge nodes
                    self.vehicles[vehicle_type][i].available_tasks = [(i,j) for j in discharge_nodes]

    def get_available_tasks(self):
        available = {}
        for vehicle_type in self.vehicles:
            available[vehicle_type]={}
            for i in range(len(self.vehicles[vehicle_type])):
                available[vehicle_type][i] = self.vehicles[vehicle_type][i].available_tasks
        return available