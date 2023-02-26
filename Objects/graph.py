import networkx as nx
import numpy as np

def get_time_to_mine_entrance():
    return np.random.expo(10)

def get_time_from_mine_entrance():
    return np.random.expo(25)


class Graph():
    
    def __init__(self,plants,stockpiles,benches,times=[],precedence = []):

        #self.route_nodes = route_nodes
        
        self.plants = plants
        self.stockpiles = stockpiles
        self.benches = benches
        self.graph = nx.Graph()
        create_graph(times)

        
    def create_graph(self,times):

        num_nodes = len(self.route_nodes)+len(self.plants)+len(self.stockpiles)+len(self.benches)+1
        
        labels = []
        # labels.append(["route_"+str() for i in range(len(self.route_nodes))])
        labels.append(["plant_"+str() for i in range(len(self.route_nodes))])
        labels.append(["stockpile_"+str() for i in range(len(self.route_nodes))])
        labels.append(["benches_"+str() for i in range(len(self.route_nodes))])
        labels.append("mine_entrance")
        

        if len(times) == 0:
            
            #create nodes_:
            nodes = np.arange(self.num_nodes)
            self.graph.add_nodes = nodes
            #add name to nodes:
            nx.set_node_attributes(self.graph, labels, "labels")
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
            
            self.graph.add_eges_from(edges)
            #add time attribute
            nx.set_edge_attributes(self.graph,times,"times")

        else:
            pass
