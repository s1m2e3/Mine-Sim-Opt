import torch
import numpy as np
import pandas as pd
import random
from itertools import combinations
from itertools import product

class Orchestrator():
    def __init__(self,network,input_size,action_size,nodes=16,population_size=20,crossover_type="statistical" ,mutate_factor=0.9,crossover_factor=0.9,\
        random_decay = 0.001,parent_pairs=6):
        
        self.network = network
        self.graph = self.network.graph
        self.input_size = input_size
        
        self.action_size = action_size
        self.number_of_nodes = nodes
        self.nodes = nodes 
        self.population_size = population_size
        self.crossover_type = "1_point"
        self.optim_type = "ELM"
        self.mutate_factor = 0.9
        self.crossover_factor = 0.9
        self.random_decay = 0.0001
        self.agents = [Agent(self.input_size,self.action_size,self.nodes) for i in range(self.population_size)]
        self.parents_pair = parent_pairs
    
    def evaluate_pop(self):
        fitness = {}
        rewards = []
        novelty = []
        for i in range(len(self.agents)):
            fitness[i] = self.evaluate_agent(self.agents[i])
            rewards.append(fitness[i]["rewards"])
            novelty.append(fitness[i]["novelty"])
        print(sum(rewards)/len(rewards),"average rewards")
        print(sum(novelty)/len(novelty),"average novelty")
        elites = self.get_elite(fitness)
        return elites
    
    def evaluate_agent(self,agent,mop="pareto"):
        
        accum_rewards = sum(agent.memory["reward"])
        #sparsity = sum([len((agent.model[layer][0]==0).nonzero()) for layer in agent.model])
        symmetry = sum([(torch.norm(agent.model[layer][0].flatten()-torch.transpose(agent.model[layer][0],0,1).flatten())).numpy() for layer in agent.model])
        

        state = np.array(agent.memory["state"])
        action = np.array(agent.memory["action"])
        action = action.reshape(len(action),1) 
        
        
        novelty = sum(np.var(np.concatenate((state,action),axis=1),axis=1))
        
        #define all of them as favorable positive gradients 
        return {"rewards":accum_rewards,"novelty":novelty,"symmetry":symmetry}
        #return {"rewards":accum_rewards}
        #return {"rewards":accum_rewards,"sparsity":sparsity,"symmetry":symmetry,"novelty":novelty}

    def get_elite(self,fitness):
        
        #print(np.mean(fitness["rewards"])," average rewards")
        
        #M = pd.DataFrame.from_dict(fitness,orient="index").sort_values(["rewards","sparsity","symmetry","novelty"],ascending=False)
        M = pd.DataFrame.from_dict(fitness,orient="index").sort_values(["rewards","novelty","symmetry"],ascending=False).drop_duplicates()
        #M = pd.DataFrame.from_dict(fitness,orient="index").sort_values(["rewards","novelty"],ascending=False).drop_duplicates()
        print(M)
        n, m = M.shape
        N = []
        
        sorted_coefficients = np.array(M)

        # Initialize the Pareto frontier with the first element.
        pareto_frontier = [0]

        for i in range(len(sorted_coefficients)):
            dominated=False
            for j in range(len(sorted_coefficients)):
                if i != j:
                    #test if no worse
                    if np.all(sorted_coefficients[j,:]>=sorted_coefficients[i,:]):
                        dominated=True
                        break
            if not dominated:
                #test if strictly better in at least one 
                for elem in pareto_frontier: 
                    if not np.any(sorted_coefficients[i,:]>=sorted_coefficients[elem,:]):
                        break
                if i not in pareto_frontier:
                    pareto_frontier.append(i)
        front = M.iloc[pareto_frontier].index.values
        print(M.iloc[pareto_frontier])
        return front

    def mutate(self,agent):
        
    
            #if self.mutate_factor>np.random.uniform(0,1):
        if self.optim_type=="GA":     
            optim = {"GA":{}}
            layers = list(agent.model)
            layer = layers[-1]
            mutate_options = ["weight_change"]
            activation_options = ["sigmoid","tanh","relu","none"]
            mutate_option = random.choice(mutate_options) 
            optim["GA"][mutate_option]=[layer]
            all_index = agent.model[layer][0].shape
            dims = [list(np.arange(elem)) for elem in all_index]
            all_index=[]
            for element in product(*dims): 
                all_index.append(element)
            num = int(len(all_index)*0.1) 
            
            # if list(optim["GA"])[0] in ["cut","connect","weight_change"]:
                
            #     indices = random.choices(self.get_indexes(agent,all_index,k=num))
            #     
            # else:
            #     activation = random.choice(activation_options)
            #     optim["GA"][mutate_option].append(activation)
            
            for i in range(agent.output_size):
                indices = [pair for pair in all_index if pair[0]==i]
                indices = random.choices(indices,k=num)
                optim["GA"][mutate_option].append(indices)
                agent.optimization(optim)
        
        elif self.optim_type=="ELM":
        
            agent.optimization({"ELM":[]})

    def combine_statistically_weights(self,t1,t2,shape):
        
        mean_1 = torch.mean(t1,axis=1)
        std_1 = torch.std(t1,axis=1)
        mean_2 = torch.mean(t2,axis=1)
        std_2 = torch.std(t2,axis=1)
        new_mean = mean_1+mean_2
        new_std = ((std_1**2+std_2**2)/2)**(1/2)
        new = torch.zeros(shape)
        for node in range(len(torch.tensor(new_mean))):
            new[node,:]=torch.normal(torch.tensor(new_mean)[node],torch.tensor(new_std)[node],size=(1,shape[1]))
        return new

    def combine_statistically_bias(self,t1,t2,shape):

        mean_1 = torch.mean(t1)
        std_1 = torch.std(t1)
        mean_2 = torch.mean(t2)
        std_2 = torch.std(t2)

        new_mean = mean_1+mean_2
        new_std = ((std_1**2+std_2**2)/2)**(1/2)
        
        new = torch.zeros(shape)
        
        for node in range(len(new)):
            new[node]=torch.normal(torch.tensor(new_mean),torch.tensor(new_std))
        return new

    def cross_over(self,p1,p2):
        
        layer_1 = len(p1.model)
        layer_2 = len(p2.model)
        kids = [p1,p2]
        
        #test analytical crossover
        if layer_1 == layer_2:
            #if number of layers is the same, crossover all of it" 
    
            if self.crossover_type=="statistical":
                for k  in range(len(kids)):
                    for layer in range(1,layer_1+1):
                        
                        for i in [0,1]:

                            

                            if i == 0:
                                kids[k].model[layer][i]=self.combine_statistically_weights(p1.model[layer][i],p2.model[layer][i],kids[k].model[layer][i].shape )
                                
                            else:
                                #new=np.random.normal(loc = new_mean,scale= new_std)
                                kids[k].model[layer][i]=self.combine_statistically_bias(p1.model[layer][i],p2.model[layer][i],kids[k].model[layer][i].shape)
                                

                                
                
            #elif self.crossover=="analytical":
            elif self.crossover_type=="1_point":
                
                #choose layer for crossover
                layer = random.choice(list(p1.model))
                flatten_1 = torch.flatten(p1.model[layer][0])
                flatten_2 = torch.flatten(p2.model[layer][0])
                cut_index = random.randint(0,len(flatten_1))
                new_1 = torch.cat((flatten_1[0:cut_index],flatten_2[cut_index:])).reshape(p1.model[layer][0].shape)
                new_2 = torch.cat((flatten_2[0:cut_index],flatten_1[cut_index:])).reshape(p2.model[layer][0].shape)
                kids[0].model[layer][0]=new_1
                kids[1].model[layer][0]=new_2
                print("crossover done")

        elif layer_1>layer_2:
            #layer_2 is shorter
            if self.crossover_type=="statistical":
                for kid in kids:
                    for layer in range(1,layer_2+1):
                        
                        for i in [0,1]:
                            
                            

                            if i == 0:
                                kids[k].model[layer][i]=self.combine_statistically_weights(p1.model[layer][i],p2.model[layer][i],kids[k].model[layer][i].shape )
                                
                            else:
                                #new=np.random.normal(loc = new_mean,scale= new_std)
                                kids[k].model[layer][i]=self.combine_statistically_bias(p1.model[layer][i],p2.model[layer][i],kids[k].model[layer][i].shape)
                                
                                
            #elif self.crossover=="analytical":
            elif self.crossover_type=="1_point":
                
                 #choose layer for crossover
                layer = random.choice(list(p2.model))
                flatten_1 = torch.flatten(p1.model[layer][0])
                flatten_2 = torch.flatten(p2.model[layer][0])
                cut_index = random.randint(0,len(flatten_1))
                new_1 = torch.cat((flatten_1[0:cut_index],flatten_2[cut_index:])).reshape(p1.model[layer][0].shape)
                new_2 = torch.cat((flatten_2[0:cut_index],flatten_1[cut_index:])).reshape(p2.model[layer][0].shape)
                kids[0].model[layer][0]=new_1
                kids[1].model[layer][0]=new_2

        else:
            #layer 1 is shorter
            if self.crossover_type=="statistical":
                for kid in kids:
                    for layer in range(1,layer_1+1):
                        
                        for i in [0,1]:
                            
                            if i == 0:
                                kids[k].model[layer][i]=self.combine_statistically_weights(p1.model[layer][i],p2.model[layer][i],kids[k].model[layer][i].shape )
                                
                            else:
                                #new=np.random.normal(loc = new_mean,scale= new_std)
                                kids[k].model[layer][i]=self.combine_statistically_bias(p1.model[layer][i],p2.model[layer][i],kids[k].model[layer][i].shape)
                                
                                
            #elif self.crossover=="analytical":
            elif self.crossover_type=="1_point":
                 #choose layer for crossover
                layer = random.choice(list(p1.model))
                flatten_1 = torch.flatten(p1.model[layer][0])
                flatten_2 = torch.flatten(p2.model[layer][0])
                cut_index = random.randint(0,len(flatten_1))
                new_1 = torch.cat((flatten_1[0:cut_index],flatten_2[cut_index:])).reshape(p1.model[layer][0].shape)
                new_2 = torch.cat((flatten_2[0:cut_index],flatten_1[cut_index:])).reshape(p2.model[layer][0].shape)
                kids[0].model[layer][0]=new_1
                kids[1].model[layer][0]=new_2
        return kids           

    def train(self,robots_capacity,periods,max_steps=20,n_iters=10):

        warehouse = list(self.network.get_warehouse())[0]
        objectives = random.choices(list(self.graph.nodes),k=robots_capacity)
        self.set_objectives(objectives)
        
        for episode in range(n_iters):
            for i in range(len(self.agents)):
                state = warehouse
                for j in range(max_steps):
                    
                    passed_state = self.get_heuristics(state)
                    prediction = self.agents[i].predict(passed_state).numpy()
                    
                    action = np.argmax(self.agents[i].predict(passed_state).numpy())
                    new_state,reward = self.step(state,action)
                    self.agents[i].update_memory(passed_state,action,reward,self.get_heuristics(new_state))
                    self.agents[i].update_q_values(passed_state,action,reward,self.get_heuristics(new_state))
                    if new_state in self.get_objectives():
                        self.remove_objective(new_state)
                    
                        break
                    state = new_state
            print("finished steps for each agent episode: ",episode)
            parents = self.evaluate_pop()    
            #if episode % 20 ==0:
            if random.random()<self.crossover_factor:
                
                
                not_elite = [i for i in range(self.population_size) if i not in parents]
            
                if len(parents)>1:
                    parents_comb = list(combinations(parents,2))
                    
                    for pair in parents_comb:
                        kids = self.cross_over(self.agents[pair[0]],self.agents[pair[1]])
                        self.agents[pair[0]] = kids[0]
                        self.agents[pair[1]] = kids[1]
                    
                    for kid in not_elite:
                        parent = random.choice(parents)
                        
                        self.agents[kid].model=self.agents[parent].model
                        self.mutate(self.agents[kid])
                else:
                    for kid in not_elite:
                        self.agents[kid].model=self.agents[parents[0]].model
                        self.mutate(self.agents[kid])
                
                self.crossover_factor -= self.random_decay*0.01

            if  random.random()<self.mutate_factor:
                for agent in self.agents:
                    self.mutate(agent)
                
                    agent.dump_memory()
                self.mutate_factor -= self.random_decay
                

    def step(self,state,action):
        action=action+1
        #check if action is legal
        new_state = list(state)
        reward = 0
        if action==1:
            new_state[1]=new_state[1]-1
            if tuple(new_state) in list(self.graph.nodes):
                
                reward = -1
            else:
                reward = -50
                new_state = state
        elif action==2:
            new_state[0]=new_state[0]+1
            if tuple(new_state) in list(self.graph.nodes):
                
                reward = -1
            else:
                reward = -50
                new_state = state
        elif action==3:
            new_state[1]=new_state[1]+1
            if tuple(new_state) in list(self.graph.nodes):
                
                reward = -1
            else:
                reward = -50
                new_state = state
        elif action==4:
            new_state[0]=new_state[0]-1
            if tuple(new_state) in list(self.graph.nodes):
                
                reward = -1
            else:
                reward = -50
                new_state = state
        new_state = tuple(new_state)
        reward += self.check_if_objective(new_state)
        
        return new_state,reward

    def check_if_objective(self,state):
        objectives = self.get_objectives()
        if state in objectives:
            print("found reward")
            reward=+100
        else:
            reward=0
        return reward
    def set_objectives(self,goals):
        self.objectives=goals
    def get_objectives(self):
        return self.objectives
    def remove_objective(self,state):
        self.objectives.remove(state)
    def get_heuristics(self,state):
        state = list(state)
        objectives = self.get_objectives()
        expanded_states = [state]
        #check if action is legal
        
        for i in [1,2,3,4]:
            new_state = state
            for j in [2]:
                
                if i==1:
                    
                    expanded_states.append([state[0],state[1]-j])
                    new_state[1] = new_state[1]-j
                    expanded_states.append(new_state)
                    
                elif i==2:
                    expanded_states.append([state[0]+j,state[1]])
                    new_state[0] = new_state[0]+j
                    expanded_states.append(new_state)
                                                           
                    
                elif i==3:
                    expanded_states.append([state[0],state[1]+j])
                    new_state[1] = new_state[1]+j
                    expanded_states.append(new_state)
                                      
                    
                elif i==4:
                    expanded_states.append([state[0]-j,state[1]])
                    new_state[0] = new_state[0]-j
                    expanded_states.append(new_state)
        
        for i in range(len(expanded_states)):
            
            if tuple(expanded_states[i]) in list(self.graph.nodes):             
                heuristic = []
                for objective in objectives:
                    heuristic.append(self.network.heuristic_function(expanded_states[i],objective))
                expanded_states[i]=min(heuristic)
            else:
                expanded_states[i]=25

        return expanded_states       

class Agent():

    
    def __init__(self,input_size,output_size,number_of_nodes,physics_informed=False,memory_buffer_size = 64):
        
        #optimization types are GA, ELM and 
        
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_nodes = number_of_nodes
        self.model = {1:[torch.rand((self.number_of_nodes,self.input_size),dtype=torch.float,requires_grad=False),torch.randn(self.number_of_nodes,dtype=torch.float,requires_grad=False),"tanh"],2:[torch.randn((self.output_size,self.number_of_nodes),dtype=torch.float,requires_grad=False),torch.randn(self.output_size,dtype=torch.float,requires_grad=False),"none"]}
        self.buffer_size = memory_buffer_size
        self.memory = {"state":[],"action":[],"reward":[],"new_state":[]}
        self.q_values = {}
        self.discount_factor = 0.3

    def predict(self,state):
        
        output = torch.tensor(state,dtype=torch.float)
        for i in self.model:
            
            output = torch.matmul(self.model[i][0],output)
            output = self.activation_function(i,output)
            output = torch.add(output,self.model[i][1])
            
        return output

    def predict_h(self,state):
        output = torch.tensor(state,dtype=torch.float)
        for i in [1]:
            
            output = torch.matmul(self.model[i][0],output)
            output = self.activation_function(i,output)
            output = torch.add(output,self.model[i][1])
            
        return output
    def activation_function(self,index,matrix):
        
        if self.model[index][2] == "sigmoid":
            output = torch.special.expit(matrix)
        elif self.model[index][2] == "tanh":
            output = torch.tanh(matrix)
        elif self.model[index][2] == "relu":
            output = torch.relu(matrix)
        else:
            output = matrix


        return output  

    def update_memory(self,state,action,reward,new_state):

        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)
        self.memory["new_state"].append(new_state)
    def dump_memory(self,):
        self.memory = {"state":[],"action":[],"reward":[],"new_state":[]}

    def update_q_values(self,past_state,past_action,reward,new_state):
        
        if tuple(past_state) not in self.q_values:
            self.q_values[tuple(past_state)]=[0,0,0,0]
            self.q_values[tuple(past_state)][past_action]= self.discount_factor*(reward+0)
        else:
            
            self.q_values[tuple(past_state)][past_action] = (1-self.discount_factor)*self.q_values[tuple(past_state)][past_action]+ self.discount_factor*(reward+self.discount_factor*max( self.q_values[tuple(past_state)]))

    def optimization(self,optim):   
        #currently not working for biases
        for optim_type in optim:
        
            if optim_type=="GA":
                command = list(optim[optim_type])[0]
                
                    #define cuts in weights ,activation of weights, weight change and  
                    #optim will be a dictionary having {"GA":{"cut":[layer,indices],"activate":[layer,indices],"weight_change":[layer,indices],
                    # "add_layer":"activation_function","change_function":[layer,"activation"]},"gradient":[learning_rate]}
                if "cut" in command:
                    sub_list = int(len(optim[optim_type][command][1])*0.1)
                    sub_list = random.choices(optim[optim_type][command][1],k=sub_list)
                    for index in sub_list:
                        self.model[optim[optim_type][command][0]][0][index]=0
                elif "weight_change" in command:
                    
                    change = random.uniform(-1,1)
                    
                    for index in optim[optim_type][command][1]:
                        self.model[optim[optim_type][command][0]][0][index]=self.model[optim[optim_type][command][0]][0][index]+change
                        self.model[optim[optim_type][command][0]][1][index[0]]=self.model[optim[optim_type][command][0]][1][index[0]]+change
                    
                elif "change_function" in command:
                    
                    self.model[optim[optim_type][command][0]][2]=optim[optim_type][command][1]
                elif "add_layer" in command: 
                    last_layer = list(self.model)[-1]
                    self.model[last_layer+1] = self.model[last_layer]
                    self.model[last_layer][0] = torch.randn(self.number_of_nodes,self.number_of_nodes)
                    self.model[last_layer][1] = torch.randn(self.number_of_nodes)
                    self.model[last_layer][2] = optim[optim_type][command]
            
            elif optim_type=="ELM":
                print(self.predict_errors())
                states= list(self.q_values)
                h= self.predict_h(states[0])
                for state in states[1:]:
                    
                    h=torch.cat((h,self.predict_h(state)),0)
                
                h=h.reshape((self.number_of_nodes,len(states)))
                y=torch.tensor(self.get_errors(),dtype=torch.float)
                pinv = torch.tensor(torch.linalg.pinv(h),dtype=torch.float)
                new_w=torch.matmul(torch.linalg.pinv(h),y)
                self.model[2][0]=new_w

                print(self.predict_errors())

    def get_errors(self):
        error = []
        for pair in self.q_values:
            error.append(np.array(self.q_values[pair]))
        return error

    def predict_errors(self):
        error = []
        for pair in self.q_values:
            
            error.append(np.mean(np.array(self.q_values[pair]-np.array(self.predict(pair)))))
        return sum(error)
        
        # states= self.memory["state"]
        # used_actions= self.memory["action"]
        # desired_actions = [state.index(min(state))  for state in states]
        
        # for i in range(len(desired_actions)):
        #     if desired_actions[i] in [1,2]:
        #         desired_actions[i]=0
        #     elif desired_actions[i] in [3,4]:
        #         desired_actions[i]=1
        #     elif desired_actions[i] in [5,6]:
        #         desired_actions[i]=2
        #     elif desired_actions[i] in [7,8]:
        #         desired_actions[i]=3
        # error = []
        # for i in range(len(desired_actions)):
        #     if desired_actions[i] == used_actions[i]:
        #         error.append(0)
        #     elif desired_actions[i]==0 and used_actions[i]== 2:
        #         error.append(-3)
        #     elif desired_actions[i]==0 and used_actions[i]== 1| used_actions[i]== 3:
        #         error.append(-1)
        #     elif desired_actions[i]==1 and used_actions[i]== 3:
        #         error.append(-3)
        #     elif desired_actions[i]==1 and used_actions[i]== 0| used_actions[i]== 2:
        #         error.append(-1)
        #     elif desired_actions[i]==2 and used_actions[i]== 0:
        #         error.append(-3)
        #     elif desired_actions[i]==2 and used_actions[i]== 1| used_actions[i]== 3:
        #         error.append(-1)
        #     elif desired_actions[i]==3 and used_actions[i]== 1:
        #         error.append(-3)
        #     elif desired_actions[i]==3 and used_actions[i]== 0| used_actions[i]== 2:
        #         error.append(-1)
        

        # returns y