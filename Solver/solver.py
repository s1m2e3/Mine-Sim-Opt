import numpy as np
import torch

class Orchestrator():
    def __init__(self,graph):
        
        self.graph = graph
        self.env = self.graph.env
        self.inputs = self.env.observation_space
        self.actions = self.env.action_space
        self.agents = self.split("vehicle")

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

    # def call_analytics(self):
    #     if 
    
    


class Agent():
    
    def __init__(self,input_size,output_size,physics_informed=False,memory_buffer_size = 64):
        
        #optimization types are GA, ELM and 
        
        self.input_size = input_size
        self.output_size = output_size
        self.model = {1:[torch.rand(self.output_size,self.input_size),torch.randn(self.output_size),"sigmoid"]}
        self.buffer_size = memory_buffer_size
        self.state_memory_buffer = np.zeros(self.buffer_size,self.input_size)
        self.q_values = np.zeros(self.input_size,self.output_size)
        self.discout_factor = 0.3

    def predict(self,state):
        
        output = state
        for i in self.model:
            output = torch.matmul(output,self.model[i][0])
            output = self.activation_function(i,output)
            output = torch.add(output,self.model[i][1])
        return output

    def activation_function(self,index,matrix):
        
        if self.model[index][2] == "sigmoid":
            output = torch.special.expit(matrix)
        return output  

    def update_state_memory_buffer(self,state_matrix):

        prev  = self.state_memory_buffer[1:,:] 
        self.state_memory_buffer[:-1,:]=prev
        self.state_memory_buffer[-1,:]=state_matrix

    def update_q_values(self,past_state,past_action,reward,new_state):
        self.q_values[past_state,past_action] = (1-self.discount_factor)*self.q_values[past_state,past_action]+ self.discount_factor*(reward+self.discount_factor*max(self.q_values[new_state,:]))

    def optimization(self,optim):
        
        for optim_type in optim:
        
            if optim_type=="GA":
                
                for command in optim[optim_type]:
                    #define cuts in weights ,activation of weights, weight change and  
                    #optim will be a dictionary having {"GA":{"cut":[layer,indices],"activate":[layer,indices],"weight_change":[layer,indices]},"gradient":[learning_rate]}
                    if "cut" in optim:
                        self.model[optim[optim_type][command][0]][optim[optim_type][1]]=0
                    elif "connect" in optim:
                        self.model[optim[optim_type][command][0]][optim[optim_type][1]]=1
                    elif "weight_change" in optim:
                        for index in optim[optim_type][1]:
                            self.model[optim[optim_type][command][0]][index]=torch.rand(1)

            elif optim_type=="gradient":

                transitions = memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = policy_net(state_batch).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                # Compute Huber loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()
