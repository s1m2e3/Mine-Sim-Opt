from Objects import graph
from Objects import stockpile
from Objects import plant
from Objects import bench
from Objects import vehicle
from Solver import solver
import numpy as np
import simpy


if __name__=="__main__":

    env = simpy.Environment()
    n_plants = 2
    plants = [plant.Plant(env) for i in range(n_plants)]
    n_stockpiles = 3
    stockpiles = [stockpile.Stockpile(env,3000) for i in range(n_stockpiles)]
    n_benches = 6
    benches = [bench.Bench(3000) for i in range(n_benches)]
    mine = graph.Graph(env,plants,stockpiles,benches)
    
    n_trucks = 40
    n_shovels = 4
    
    Trucks = [vehicle.Truck(mine,60) for truck in np.arange(n_trucks)]
    Shovels = [vehicle.Shovel(mine,20) for shovel in np.arange(n_shovels)]
    
    for truck in Trucks:
        mine.add_vehicles(("Trucks",truck))
    for shovel in Shovels:
        mine.add_vehicles(("Shovels",shovel))

    num_runs = 5000
    Orchestrator = solver.Orchestrator(mine)
    # for i in num_runs:
        
    #     mine.simulate()
