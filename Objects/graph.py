import networkx as nx
import numpy as np
import simpy 

class Graph():
    
    def __init__(self,env,plants,stockpiles,benches,times=[],precedence = []):

        #self.route_nodes = route_nodes
        
        self.plants = plants
        self.stockpiles = stockpiles
        self.benches = benches
        self.graph = nx.Graph()
        self.env = env
        self.create_graph(times)
        self.num_nodes = 0
        self.vehicles = {"Trucks":[],"Shovels":[]}
        self.load_nodes = []
        self.discharge_nodes = []


    def create_graph(self,times):

        self.num_nodes = len(self.plants)+len(self.stockpiles)+len(self.benches)+1
        
        labels = []
        # labels.append(["route_"+str() for i in range(len(self.route_nodes))])
        labels.extend(["plant_"+str(i) for i in range(len(self.plants))])
        labels.extend(["stockpile_"+str(i) for i in range(len(self.stockpiles))])
        labels.extend(["benches_"+str(i) for i in range(len(self.benches))])
        labels.append("mine_entrance")
        

        if len(times) == 0:
            
            #create nodes_:
            nodes = np.arange(self.num_nodes)
            self.graph.add_nodes_from(nodes)
            #add name to nodes:
            attr_dict = {}
            for node in nodes:
                attr_dict[node]={"labels":labels[node]}
            
            nx.set_node_attributes(self.graph, attr_dict)
            labels = nx.get_node_attributes(self.graph,"labels")
            
            #nodes_routes = [i for i in range(len(labels)) if "route" in labels[i]]
            nodes_plant = [i for i in range(len(labels)) if "plant" in labels[i]]
            nodes_stockpile = [i for i in range(len(labels)) if "stockpile" in labels[i]]
            nodes_benches = [i for i in range(len(labels)) if "benches" in labels[i]]
            nodes_mine_entrance = [i for i in range(len(labels)) if "mine_entrance" in labels[i]]

            #create edges_:
            # for simple case all are connected to mine_entrance
            edges = []
            times = []
            for entrance in nodes_mine_entrance:
                
                #for route in nodes_routes:
                #    edges.append((entrance,route))
                for plant in nodes_plant:
                    edges.append((entrance,plant))
                    times.append(get_time_to_mine_entrance())
                    
                for stockpile in nodes_stockpile:
                    edges.append((entrance,stockpile))
                    times.append(get_time_to_mine_entrance())
                for benches in nodes_benches:
                    edges.append((entrance,benches))
                    times.append(get_time_from_mine_entrance())
            
            self.graph.add_edges_from(edges)
            times = dict(zip(edges,times))
            #add time attribute
            attr_dict = {}
            for pair in edges:
                attr_dict[pair]={"times":times[pair]}
            #print(attr_dict)
            nx.set_edge_attributes(self.graph,attr_dict)
            self.load_nodes = [i for i in labels if "benches" in labels[i]]
            self.discharge_nodes = [i for i in labels if "plant" or "stockpile" in labels[i]]
            self.possible_combinations = {}
            for load_node in self.load_nodes:
                self.possible_combinations[load_node]=[(load_node,discharge_node) for discharge_node in self.discharge_nodes]
            for discharge_node in self.discharge_nodes:
                self.possible_combinations[discharge_node]=[(discharge_node,load_node) for load_node in self.load_nodes]

        else:
            pass

    def add_vehicles(self,vehicle_tuple):
        self.vehicles[vehicle_tuple[0]].append(vehicle_tuple[1])

    def get_states(self):
        
        plants = [plant.resource.count for plant in self.plants]
        plants.extend([plant.processed_tonnage for plant in self.plants])
        plants.extend([len(plant.resource.queue) for plant in self.plants])
        
        benches = [bench.resource.count for bench in self.benches]
        benches.extend([bench.current_tonnage for bench in self.benches])
        benches.extend([len(bench.resource.queue) for bench in self.benches])

        stockpiles = [stockpile.resource.count for stockpile in self.stockpiles]
        stockpiles.extend([stockpile.current_tonnage for stockpile in self.stockpiles])
        stockpiles.extend([len(stockpile.resource.queue) for stockpile in self.stockpiles])

        trucks = [truck.status for truck in self.vehicles["Trucks"]]
        trucks.extend([truck.node for truck in self.vehicles["Trucks"]])
        shovels = [shovel.status for shovel in self.vehicles["Shovels"]]
        shovels.extend([shovel.node for shovel in self.vehicles["Shovels"]])

        return {"plants":plants,"benches":benches,"stockpiles":stockpiles,"trucks":trucks,"shovels":shovels}


    def step(self,action):

        for 
        pass


def get_time_to_mine_entrance():
    return np.round(np.random.exponential(10),2)

def get_time_from_mine_entrance():
    return np.round(np.random.exponential(25),2)

