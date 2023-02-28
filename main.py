from Objects import graph
from Objects import stockpile
from Objects import plant
from Objects import bench
from Objects import vehicle
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
    