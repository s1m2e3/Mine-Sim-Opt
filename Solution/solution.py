import networkx as nx
import random 
import numpy as np
class Solution():
    def __init__(self,graph):
        #vehicles is a dictionary that contains a list of Vehicle objects
        self.graph = graph
        self.vehicles = self.graph.vehicles
        self.solution = {}
        self.update_available_tasks()

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

    def update_available_tasks(self):
        
        labels = nx.get_node_attributes(self.graph.graph,"labels")
        load_nodes = self.graph.load_nodes
        discharge_nodes = self.grpah.discharge_nodes


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
    
    def from_solution_to_actions(self):
        #assuming the action space vehicles x actions
        #assuming that trucks are indexed before shovels and shovels before drilling and drilling before blasting teams
        rows = sum([len(self.vehicles[vehicle_type]) for vehicle_type in self.vehicles])
        columns = len(self.graph.possible_combinations)
        count = 0 
        actions = np.zeros((rows,columns))
        for vehicle_type in self.solution:
            for i in self.solution[vehicle_type]:
                index = self.graph.possible_combinatios.index(self.solution[vehicle_type][i])
                actions[count,index]=1
                count+=1
        del count
        return actions


    def from_actions_to_solution(self,actions):
        
        vehicle_type = list(self.vehicles)
        vehicle_number = [len(self.vehicles[vehicle_type]) for vehicle_type in self.vehicles]
        vehicle_number = [sum(vehicle_number[0:i+1]) for i in range(len(vehicle_number))]
        
        for i in range(len(actions)):
            index = np.where(actions[i,:]==1)[0]
            vehicle_solution = self.graph.possible_combinations[index]
            index_vehicle = np.where(vehicle_number>=i)[0][0]
