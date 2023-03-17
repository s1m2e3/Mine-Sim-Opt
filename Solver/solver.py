import numpy as np
import pandas as pd
import torch
import random
from itertools import product

class Orchestrator():
    def __init__(self,graph):
        
        self.graph = graph
        self.env = self.graph.env
        self.inputs = self.env.observation_space
        self.actions = self.env.action_space
        self.agents = self.split("vehicle")
        self.mutate_factor = 0.9
        self.crossover_factor = 0.9
        self.random_decay = 0.001
        self.number_of_nodes = 256
        self.crossover_type = "statistical"

    def split(self,split_type="vehicle",physics_informed=False):

        #there exist 2 types of split by vehicle and by length of inputs and actions: "vehicle","divide" respectively
        agents = {} 
        if split_type=="each":
    
            n_agents = self.actions.shape[0]
            output_size = self.actions.shape[1]
            num_agents = np.arange(n_agents)
            for i in num_agents:
                agents[i] = Agent(len(self.inputs),output_size,physics_informed)
        elif split_type=="vehicle":
            
            n_agents = len(self.graph.vehicles)
            for i in range(len(self.graph.vehicles)):
                output_size = (len(self.graph.vehicles[i]),self.actions.shape[1])
                agents[i]= Agent(len(self.inputs),output_size,physics_informed)
        else:
            num_agents = 0
            for i in [num_agents]:
                agents[i] = Agent(len(self.inputs),self.actions.shape,physics_informed)
        agents[0] = Agent(len(self.inputs),self.actions.shape,physics_informed)
        
        return agents


    def predict(self,state):
        
        action = self.ensemble(state)
        return action

    def ensemble(self,state):
        
        pred = {}
        for agent in agents:
            pred[agent]=agents[agent].predict(state)
        pred =  np.concatenate(list(pred.values()))
        return pred

    
    def evaluate_pop(self):
    
        fitness = {}
        for i in self.agents:
            fitness[i] = self.evaluate_agent(self.agents[i])
        elites = self.get_elite(fitness)
        return elites
    
    def evaluate_agent(self,agent,mop="pareto"):
        
        accum_rewards = sum(agent.memory["reward"])
        sparsity = sum([len((agent.model[layer][0]==0).nonzero()) for layer in agent.model])
        symmetry = sum([torch.norm(agent.model[layer][0].flatten()-torch.transpose(agent.model[layer][0]).flatten()) for layer in agent.model])
        novelty = sum(np.var(np.concatenate(np.array(agent.memory["state"],np.array(agent.memory["actions"]))),axis=1))
        
        #define all of them as favorable positive gradients 
        return {"rewards":accum_rewards,"sparsity":sparsity,"symmetry":symmetry,"novelty":novelty}

    def get_elite(self,fitness):
        
        M = np.array(pd.DataFrame.from_dict(fitness,orient="columns"))
        n, m = M.shape
        N = []
        for i in range(n):
            c = 0
            for j in range(n):
                if all(M[i] <= M[j]):
                    c += 1
            if c == n:
                N.append(M[i])
        return N
    
    def mutate(self):
        
        for agent in self.agents:
            if self.mutate_factor>np.random.uniform(0,1):
            
                optim = {"GA":{}}
                layers = list(agent.model)
                layer = random.choice(layers)
                mutate_options = ["cut","connect","weight_change","change_function"]
                activation_options = ["sigmoid","tanh","relu","none"]
                mutate_option = random.choice(mutate_options) 
                optim["GA"][mutate_option]=[layer]
                all_index = agent.model[layer][0].shape
                dims = [np.arange(elem) for elem in all_index]
                all_index = list(product(dims))
                num = int(self.mutate_factor**2*np.prod([elem for elem in all_index])) 
                
                if list(optim["GA"])[0] in ["cut","connect","weight_Change"]:
                    
                    indices = random.choices(all_index,k=num)
                    optim["GA"][mutate_option].append(indices)

                else:
                    activation = random.choice(activation_options)
                    optim["GA"][mutate_option].append(activation)
                
                agent.optimization(optim)
                self.mutate_factor -= self.random_decay

            else:
                pass
    
    def cross_over(self,p1,p2):
        
        layer_1 = len(p1.model)
        layer_2 = len(p2.model)
        kids = [p1,p2]
        #test analytical crossover
        if layer_1 == layer_2:
            #if number of layers is the same, crossover all of it" 
    
            if self.crossover=="statistical":
                for kid in kids:
                    for layer in p1.model:
                        for i in [0,1]:
                            if i == 0:

                                mean_1 = torch.mean(p1.model[layer][0],axis=1)
                                std_1 = torch.std(p1.model[layer][0],axis=1)
                                mean_2 = torch.mean(p2.model[layer][0],axis=1)
                                std_2 = torch.std(p2.model[layer][0],axis=1) 
                                kid.model[layer][i]=torch.tensor(np.normal(loc = mean_1+mean_2,scale= ((std_1**2+std_2**2)/2)**(1/2),size=(p1.number_of_nodes,p1,number_of_nodes)))
                                
                            else:
                                
                                mean_1 = torch.mean(p1.model[layer][1])
                                var_1 = torch.var(p1.model[layer][1])
                                mean_2 = torch.mean(p2.model[layer][1])
                                var_2 = torch.var(p2.model[layer][1])
                                kid.model[layer][i]=torch.tensor(np.normal(loc = mean_1+mean_2,scale= ((std_1**2+std_2**2)/2)**(1/2),size=(p1.number_of_nodes)))
                    
            #elif self.crossover=="analytical":
            elif self.crossover=="1_point":
                
                layers_to_cross = np.floor(len(list(p1.model))/2)
                #choose layers from p1
                layers_from_1 = random.choice(list(p1.model),k=layers_to_cross)
                layers_from_2 = list(p1.model).remove(layers_from_1)
                for layer in layers_from_2:
                    kids[0].model[layer] = p2.model[layer]
                for layer in layers_from_1:
                    kids[1].model[layer] = p1.model[layer]
        
        elif layer_1>layer_2:
            #layer_2 is shorter
            if self.crossover=="statistical":
                for kid in kids:
                    for layer in p2.model:
                        for i in [0,1]:
                            if i == 0:

                                mean_1 = torch.mean(p1.model[layer][0],axis=1)
                                std_1 = torch.std(p1.model[layer][0],axis=1)
                                mean_2 = torch.mean(p2.model[layer][0],axis=1)
                                std_2 = torch.std(p2.model[layer][0],axis=1) 
                                kid.model[layer][i]=torch.tensor(np.normal(loc = mean_1+mean_2,scale= ((std_1**2+std_2**2)/2)**(1/2),size=(p1.number_of_nodes,p1,number_of_nodes)))
                                
                            else:
                                
                                mean_1 = torch.mean(p1.model[layer][1])
                                var_1 = torch.var(p1.model[layer][1])
                                mean_2 = torch.mean(p2.model[layer][1])
                                var_2 = torch.var(p2.model[layer][1])
                                kid.model[layer][i]=torch.tensor(np.normal(loc = mean_1+mean_2,scale= ((std_1**2+std_2**2)/2)**(1/2),size=(p1.number_of_nodes)))
                    
            #elif self.crossover=="analytical":
            elif self.crossover=="1_point":
                
                layers_to_cross = np.floor(len(list(p2.model))/2)
                #choose layers from p1
                layers_from_1 = random.choice(list(p2.model),k=layers_to_cross)
                layers_from_2 = list(p2.model).remove(layers_from_1)
                for layer in layers_from_2:
                    kids[0].model[layer] = p2.model[layer]
                for layer in layers_from_1:
                    kids[1].model[layer] = p1.model[layer]

        else:
            #layer 1 is shorter
            if self.crossover=="statistical":
                for kid in kids:
                    for layer in p1.model:
                        for i in [0,1]:
                            if i == 0:

                                mean_1 = torch.mean(p1.model[layer][0],axis=1)
                                std_1 = torch.std(p1.model[layer][0],axis=1)
                                mean_2 = torch.mean(p2.model[layer][0],axis=1)
                                std_2 = torch.std(p2.model[layer][0],axis=1) 
                                kid.model[layer][i]=torch.tensor(np.normal(loc = mean_1+mean_2,scale= ((std_1**2+std_2**2)/2)**(1/2),size=(p1.number_of_nodes,p1,number_of_nodes)))
                                
                            else:
                                
                                mean_1 = torch.mean(p1.model[layer][1])
                                var_1 = torch.var(p1.model[layer][1])
                                mean_2 = torch.mean(p2.model[layer][1])
                                var_2 = torch.var(p2.model[layer][1])
                                kid.model[layer][i]=torch.tensor(np.normal(loc = mean_1+mean_2,scale= ((std_1**2+std_2**2)/2)**(1/2),size=(p1.number_of_nodes)))
                    
            #elif self.crossover=="analytical":
            elif self.crossover=="1_point":
                layers_to_cross = np.floor(len(list(p1.model))/2)
                #choose layers from p1
                layers_from_1 = random.choice(list(p1.model),k=layers_to_cross)
                layers_from_2 = list(p1.model).remove(layers_from_1)
                for layer in layers_from_2:
                    kids[0].model[layer] = p2.model[layer]
                for layer in layers_from_1:
                    kids[1].model[layer] = p1.model[layer]
        return kids           



class Agent():
    
    def __init__(self,input_size,output_size,number_of_nodes,physics_informed=False,memory_buffer_size = 64):
        
        #optimization types are GA, ELM and 
        
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_nodes = number_of_nodes
        self.model = {1:[torch.rand(self.number_of_nodes,self.input_size),torch.randn(self.number_of_nodes),"sigmoid"],2:[torch.randn(self.output_size,self.number_of_nodes),torch.randn(self.output_size),"none"]}
        self.buffer_size = memory_buffer_size
        self.memory = {"state":[],"action":[],"reward":[],"new_state":[]}
        
        self.q_values = np.zeros(self.input_size,self.output_size)
        self.discout_factor = 0.3

    def predict(self,state):
        
        output = state
        for i in self.model:
            output = torch.matmul(self.model[i][0],output)
            output = self.activation_function(i,output)
            output = torch.add(self.model[i][1],output)
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

    def update_q_values(self,past_state,past_action,reward,new_state):
        self.q_values[past_state,past_action] = (1-self.discount_factor)*self.q_values[past_state,past_action]+ self.discount_factor*(reward+self.discount_factor*max(self.q_values[new_state,:]))

    def optimization(self,optim):   
        
        for optim_type in optim:
        
            if optim_type=="GA":
                
                for command in optim[optim_type]:
                    #define cuts in weights ,activation of weights, weight change and  
                    #optim will be a dictionary having {"GA":{"cut":[layer,indices],"activate":[layer,indices],"weight_change":[layer,indices],
                    # "add_layer":"activation_function","change_function":[layer,"activation"]},"gradient":[learning_rate]}
                    if "cut" in command:
                        self.model[optim[optim_type][command][0]][optim[optim_type][1]]=0
                    elif "connect" in command:
                        self.model[optim[optim_type][command][0]][optim[optim_type][1]]=1
                    elif "weight_change" in command:
                        for index in optim[optim_type][1]:
                            self.model[optim[optim_type][command][0]][index]=torch.rand(1)
                    elif "change_function" in command:
                        self.model[optim[optim_type][command][0]][2]=optim[optim_type][command][1]
                    elif "add_layer" in command: 
                        last_layer = list(self.model)[-1]
                        self.model[last_layer+1] = self.model[last_layer]
                        self.model[last_layer][0] = torch.randn(self.number_of_nodes,self.number_of_nodes)
                        self.model[last_layer][1] = torch.randn(self.number_of_nodes)
                        self.model[last_layer][2] = optim[optim_type][command]
            # elif optim_type=="gradient":

            #     transitions = memory.sample(BATCH_SIZE)
            #     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            #     # detailed explanation). This converts batch-array of Transitions
            #     # to Transition of batch-arrays.
            #     batch = Transition(*zip(*transitions))

            #     # Compute a mask of non-final states and concatenate the batch elements
            #     # (a final state would've been the one after which simulation ended)
            #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            #                                         batch.next_state)), device=device, dtype=torch.bool)
            #     non_final_next_states = torch.cat([s for s in batch.next_state
            #                                                 if s is not None])
            #     state_batch = torch.cat(batch.state)
            #     action_batch = torch.cat(batch.action)
            #     reward_batch = torch.cat(batch.reward)

            #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            #     # columns of actions taken. These are the actions which would've been taken
            #     # for each batch state according to policy_net
            #     state_action_values = policy_net(state_batch).gather(1, action_batch)

            #     # Compute V(s_{t+1}) for all next states.
            #     # Expected values of actions for non_final_next_states are computed based
            #     # on the "older" target_net; selecting their best reward with max(1)[0].
            #     # This is merged based on the mask, such that we'll have either the expected
            #     # state value or 0 in case the state was final.
            #     next_state_values = torch.zeros(BATCH_SIZE, device=device)
            #     with torch.no_grad():
            #         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            #     # Compute the expected Q values
            #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            #     # Compute Huber loss
            #     criterion = nn.SmoothL1Loss()
            #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            #     # Optimize the model
            #     optimizer.zero_grad()
            #     loss.backward()
            #     # In-place gradient clipping
            #     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            #     optimizer.step()
