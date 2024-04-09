import numpy as np
import pandas as pd
from tqdm import tqdm
from SAC.Agent import Agent
from Robot.Environment import Env
import matplotlib.pyplot as plt
from IPython import display
import time
import copy
import torch
import cv2

class Agent_Training():
    def __init__(self, subpolicy, image_dims, seed = 1):
        self.seed = seed
        self.image_dims = image_dims
        self.subpolicy = subpolicy
        self.static = False
        self.n_epochs = 250
        self.n_eval_episodes = 5
        self.n_test_episodes = 15
        self.episode_len = 100
        self.exploration_steps = 1000
        self.seq_len = 4
        
        self.env = Env(self.image_dims, self.seed)
        self.n_actions = self.env.action_space.shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent = Agent(self.image_dims, self.n_actions, self.seq_len, self.env, self.seed) 
            
        if self.subpolicy > 1: self.trained_actors = self.get_trained_actors(target_subpolicy = self.subpolicy)

            
    """
    REMEMBER - WE PROCESS THE IMAGES INSIDE THE ENV CLASS, WHERE WE CONCATENATE THE TWO IMAGES (FRONT AND LEFT CAMERA) 
    AS 6 CHANNELS, RESHAPE TO (CHANNEL, HEIGHT, WEIDTH), AND THEN NORMALIZE (IMAGES / 255) - see: def my_render(image_dims)
    """
            
        #### function to 
    def integrate_weights(self):
        """
        - Specify which layers to transfer, and which to freeze
        """
        pass
    
    def get_tensor(self, observation):
        observation = torch.Tensor(observation).unsqueeze(0).to(self.device)
        return observation
    
    def get_trained_actors(self, target_subpolicy):
        """
        - Get the trained actor either to test, or to get to the initial state of the next subpolicy
        """
        trained_actors = []
        for i in range(1, target_subpolicy):
            new_agent = Agent(self.image_dims, self.n_actions, self.seq_len, self.env, self.seed) 
            trained_actor = new_agent.actor
            trained_actor.load_state_dict(torch.load(f'Models/models_sub_{i}/Actor'))
            trained_actor.eval()
            trained_actors.append(trained_actor)
        return trained_actors
    
    def update_sequences(self, seq_observation, seq_action, observation, actions):
        seq_observation = np.roll(seq_observation, -1, axis=0)
        seq_observation[-1] = observation
        seq_action = np.roll(seq_action, -1, axis=0)
        seq_action[-1] = actions
        return seq_observation, seq_action
    
    @torch.no_grad()
    def sample_actions(self, seq_observation, seq_action, actor, reparameterize):
        action, _ = actor.sample_normal(self.get_tensor(seq_observation), self.get_tensor(seq_action[1:]), reparameterize=reparameterize)
        action = action.cpu().detach().numpy()[0]
        return action
    
    def initial_window(self):
        """
        - Get the initial observation sequence for first subpolicy
        - We use neutral actions (0) to minimize the interactions with env
        """
        observation = self.env.reset()
        seq_observation = []
        seq_observation_ = []
        seq_actions = []        
        for i in range(self.seq_len): 
            subpolicy = 1
            action = [0 for i in range(self.n_actions)]
            observation_, _, done, _ = self.env.step(action, subpolicy, self.static)
            seq_observation.append(observation)
            seq_observation_.append(observation_)
            seq_actions.append(action)
            observation = observation_
        return np.array(seq_observation), np.array(seq_observation_), np.array(seq_actions, dtype=np.float64)
    

    def initial_window_all(self, subpolicy=None, train=False):
        """
        - Get the initial observation sequence for all subpolicy
        """
        if subpolicy is None: subpolicy = self.subpolicy

        if subpolicy == 1:
            final_seq_observation, final_seq_observation_, final_seq_actions = self.initial_window()
        else:
            succesfull = False
            for trained_subpolicy in range(1, subpolicy):
                trained_actor = self.trained_actors[trained_subpolicy - 1]            
                while not succesfull:
                    if trained_subpolicy == 1: seq_observation, _, seq_actions = self.initial_window()
                    seq_observation_ = copy.deepcopy(seq_observation)
                    last_seq_observation = []
                    last_seq_observation_ = []
                    last_seq_actions = []
                    for t in range(self.episode_len):
                        action = self.sample_actions(seq_observation, seq_actions, trained_actor, reparameterize=False)
                        observation_, _, done, succesfull = self.env.step(action, trained_subpolicy, self.static)
                        seq_observation_, seq_actions = self.update_sequences(seq_observation_, seq_actions, observation_, action)
                        last_seq_observation.append(seq_observation)
                        last_seq_observation_.append(seq_observation_)
                        last_seq_actions.append(seq_actions)
                        seq_observation = seq_observation_
                        if done: break
            final_seq_observation ,final_seq_observation_ , final_seq_actions = last_seq_observation[-1], last_seq_observation_[-1], last_seq_actions[-1]
        if train:
            for s in range(0, self.seq_len - 1):
                self.agent.remember(final_seq_observation[s], final_seq_actions[s], None, final_seq_observation_[s], 0)
            
        return final_seq_observation, final_seq_actions

    
    def initial_exploration(self):
        """
        - Explore the environomnt to fill in the buffer and exceed the batch size
        """
        seq_observation, seq_action = self.initial_window_all()        
        seq_observation_ = copy.deepcopy(seq_observation)
        for t in tqdm(range(self.exploration_steps)):
            action = self.sample_actions(seq_observation, seq_action, self.agent.actor, reparameterize=True)
            observation_, reward, done, _ = self.env.step(action, self.subpolicy, self.static)
            seq_observation_, seq_action = self.update_sequences(seq_observation_, seq_action, observation_, action)
            self.agent.remember(seq_observation[-1], seq_action[-1], reward, seq_observation_[-1], done)
            seq_observation = seq_observation_
            if done: 
                seq_observation, seq_action = self.initial_window_all()   
                seq_observation_ = copy.deepcopy(seq_observation)
        print('------------ Hey Fella, The Initial Exploration Has Just Finished -----------------')
        

    def train(self):
        """
        - Train the agent, save metrics for visualization, and save model
        """
        all_mean_rewards, all_actor_loss = [], []
        self.initial_exploration() 
        for epoch in range(self.n_epochs):
            seq_observation, seq_action = self.initial_window_all()  
            seq_observation_ = copy.deepcopy(seq_observation)
            for t in range(self.episode_len):
                action = self.sample_actions(seq_observation, seq_action, self.agent.actor,  reparameterize=True)
                observation_, reward, done, info = self.env.step(action, self.subpolicy, self.static) 
                seq_observation_, seq_action = self.update_sequences(seq_observation_, seq_action, observation_, action)
                # ----------- Store Transitions --------------
                self.agent.remember(seq_observation[-1], seq_action[-1], reward, seq_observation_[-1], done) #######  CHANGE HERE FOR BUFFER!!!
                actor_loss = self.agent.learn()
                seq_observation = seq_observation_
                if done: break

            mean_rewards = self.validate_train()
            if epoch % 20 == 0: self.agent.save_models(self.subpolicy, epoch)
            
            all_mean_rewards.append(mean_rewards)
            all_actor_loss.append(actor_loss)
            print(f'Epoch: {epoch}, Rewards: {mean_rewards}, Actor Loss: {actor_loss}')

            plt.plot(all_mean_rewards)
            plt.xlabel('Epoch')
            plt.ylabel('Mean reward')
            plt.title('Rewards over epochs')
            plt.savefig(f'plots/Rewards_{self.subpolicy}.png')  
        
        # Save data as text file
        all_data = np.array([all_mean_rewards, all_actor_loss]).astype(float)
        all_data = all_data.T
        with open(f'Data/Data_sub_{self.subpolicy}.txt', 'w') as file:
            file.write('Rewards \tActor Loss\n')  # Write column headers
            np.savetxt(file, all_data, delimiter='\t', fmt='%f')
        
        
    def validate_train(self):
        """
        - Validate the agent during training in every epoch
        """
        total_rewards = 0
        actor = self.agent.actor.eval()
        for _ in range(self.n_eval_episodes):
            episode_reward = 0
            seq_observation, seq_action = self.initial_window_all()
            seq_observation_ = copy.deepcopy(seq_observation)
            for t in range(self.episode_len):
                action = self.sample_actions(seq_observation, seq_action, actor, reparameterize=False)
                observation_, reward, done, info = self.env.step(action, self.subpolicy, self.static)
                if info: print('--------------- Succesful ---------------')
                episode_reward += reward
                seq_observation_, seq_action = self.update_sequences(seq_observation_, seq_action, observation_, action)
                seq_observation = seq_observation_
                if done: break
            total_rewards += episode_reward
        return total_rewards / self.n_eval_episodes
            
        
if __name__ == "__main__":
    A = Agent_Training(subpolicy = 1, image_dims = 64)
    A.train()