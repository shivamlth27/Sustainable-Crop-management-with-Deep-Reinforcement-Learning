
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def choose_action(self, state):
        state_values = list(state.values())
        state_tensor = torch.FloatTensor(state_values).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        action = q_values.argmax().item()
        return action
    
    def learn(self, state, action, reward, next_state, done):
        state_values = list(state.values())  # Convert dictionary values to a list
        next_state_values = list(next_state.values())  # Convert next state values to a list

        state_tensor = torch.FloatTensor(state_values).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state_values).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        target = q_values.clone()
        target[0][action] = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)

        loss = self.loss_function(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



import pandas as pd
df= pd.read_csv(r'C:\Users\Ayush\OneDrive\Desktop\AI-search-algo\ai-project\datasets\fertilizer_recommendation.csv')

crop_mapping = {'Maize': 1, 'Paddy': 0, 'Sugarcane': 2, 'Cotton': 3, 'Tobacco': 4, 'Barley': 5, 'Wheat': 6, 'Millets': 7}
soil_mapping = {'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4}

df['Crop Type'] = df['Crop Type'].map(crop_mapping)
df['Soil Type'] = df['Soil Type'].map(soil_mapping)



class FertilizerEnvironment:
    def __init__(self, initial_state):
        self.state_size = len(initial_state)  # Include all features
        self.state = initial_state
        self.weight_crop_yield = 1.0
        self.weight_nitrogen_reward = 0.8
        self.weight_nutrient_penalty = 0.5
        self.weight_soil_reward = 0.7
        self.best_reward = float('-inf')
        self.last_action = None
        self.available_fertilizers = {
            'Urea': {'nitrogen': 46, 'phosphorous': 0, 'potassium': 0},
            'DAP': {'nitrogen': 18, 'phosphorous': 46, 'potassium': 0},
            '14-35-14': {'nitrogen': 14, 'phosphorous': 35, 'potassium': 14},
            '28-28': {'nitrogen': 28, 'phosphorous': 28, 'potassium': 0},
            '17-17-17': {'nitrogen': 17, 'phosphorous': 17, 'potassium': 17},
            '20-20': {'nitrogen': 20, 'phosphorous': 20, 'potassium': 20}
        }
        self.max_steps = 100
        self.current_step = 0
        self.crop_growth_model = {
            1 : {'nitrogen_slope': 1.5, 'phosphorous_slope': 1.0, 'potassium_slope': 0.8},
            2 : {'nitrogen_slope': 1.2, 'phosphorous_slope': 0.8, 'potassium_slope': 0.6},
            3 : {'nitrogen_slope': 1.0, 'phosphorous_slope': 0.6, 'potassium_slope': 0.5},
            4 : {'nitrogen_slope': 1.3, 'phosphorous_slope': 1.0, 'potassium_slope': 0.7},
            0 : {'nitrogen_slope': 1.4, 'phosphorous_slope': 1.2, 'potassium_slope': 0.9},
            5 : {'nitrogen_slope': 1.2, 'phosphorous_slope': 1.1, 'potassium_slope': 0.7},
            6 : {'nitrogen_slope': 1.3, 'phosphorous_slope': 1.0, 'potassium_slope': 0.8},
            7: {'nitrogen_slope': 1.1, 'phosphorous_slope': 0.9, 'potassium_slope': 0.6},
            8 : {'nitrogen_slope': 1.4, 'phosphorous_slope': 1.2, 'potassium_slope': 0.9}
        }
        self.soil_type_effects = {
            0 : {'nitrogen_effect': 0.8, 'phosphorous_effect': 0.9, 'potassium_effect': 0.7},
            1 : {'nitrogen_effect': 1.0, 'phosphorous_effect': 1.0, 'potassium_effect': 1.0},
            2 : {'nitrogen_effect': 1.2, 'phosphorous_effect': 1.1, 'potassium_effect': 1.1},
            3 : {'nitrogen_effect': 1.1, 'phosphorous_effect': 1.0, 'potassium_effect': 0.9},
            4 : {'nitrogen_effect': 0.9, 'phosphorous_effect': 1.2, 'potassium_effect': 1.3}
        }
        self.optimal_nitrogen_levels = {
            1 : 35,
            2 : 40,
            3 : 30,
            4 : 25,
            0 : 30,
            5 : 35,
            6 : 40,
            7: 30,
            8 : 25
        }

    # def step(self, action):
    #     fertilizer_name = list(self.available_fertilizers.keys())[action]
    #     selected_fertilizer = self.available_fertilizers[fertilizer_name]
    #     updated_state = self.apply_fertilizer(selected_fertilizer)
    #     crop_growth = self.simulate_crop_growth(updated_state)
    #     reward = self.calculate_reward(updated_state, crop_growth)
    #     self.state = updated_state
    #     self.current_step += 1
    #     done = self.current_step >= self.max_steps
    #     return updated_state, reward, done, {}
    
    def step(self, action):
        fertilizer_name = list(self.available_fertilizers.keys())[action]
        selected_fertilizer = self.available_fertilizers[fertilizer_name]
        updated_state = self.apply_fertilizer(selected_fertilizer)
        crop_growth = self.simulate_crop_growth(updated_state)
        reward = self.calculate_reward(updated_state, crop_growth)

        # Keep track of best fertilizer and reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_fertilizer = fertilizer_name

        self.state = updated_state
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return updated_state, reward, done, {}
    
    def reset(self):
        self.current_step = 0
        return self.state

    def apply_fertilizer(self, selected_fertilizer):
        updated_state = self.state.copy()
        for nutrient, value in selected_fertilizer.items():
            updated_state[nutrient.lower()] += value
        return updated_state

    def simulate_crop_growth(self, state):
        crop_type = state['crop_type']
        nitrogen_level = state['nitrogen']
        phosphorous_level = state['phosphorous']
        potassium_level = state['potassium']
        soil_type = state['soil_type']
        nitrogen_level *= self.soil_type_effects[soil_type]['nitrogen_effect']
        phosphorous_level *= self.soil_type_effects[soil_type]['phosphorous_effect']
        potassium_level *= self.soil_type_effects[soil_type]['potassium_effect']
        crop_params = self.crop_growth_model.get(crop_type, {})
        crop_yield = 1.0  # Base yield
        if crop_params:
            crop_yield = (18.48*(nitrogen_level * crop_params['nitrogen_slope']) +
                          15.32*(phosphorous_level * crop_params['phosphorous_slope']) +
                          14.98*(potassium_level * crop_params['potassium_slope']))
        crop_yield = min(crop_yield, 1.0)
        crop_growth = {'crop_yield': crop_yield}
        return crop_growth

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_reward(self, state, crop_growth):
        crop_yield = crop_growth['crop_yield']
        nitrogen_level = state['nitrogen']
        phosphorous_level = state['phosphorous']
        potassium_level = state['potassium']
        soil_type = state['soil_type']
        optimal_nitrogen = self.optimal_nitrogen_levels.get(state['crop_type'], 0)
        deviation = abs(nitrogen_level - optimal_nitrogen)
        nitrogen_reward = max(0, 1.0 - deviation / optimal_nitrogen)
        # if random.random() < 0.2:  # Randomly modify reward
        #     nitrogen_reward += random.uniform(-0.2, 0.2)
        phosphorous_penalty = max(0, phosphorous_level - 30) / 30.0 
        phosphorous_penalty = 1 - (1 - phosphorous_penalty) ** 2
        potassium_penalty = max(0, potassium_level - 20) / 20.0 
        potassium_penalty = 1 - (1 - potassium_level) ** 2
        nutrient_penalty = 1.0 - (phosphorous_penalty + potassium_penalty)
        soil_reward = self.soil_type_effects[soil_type]['nitrogen_effect'] + self.soil_type_effects[soil_type]['phosphorous_effect'] + self.soil_type_effects[soil_type]['potassium_effect']
        reward = (self.weight_crop_yield * crop_yield +
                  self.weight_nitrogen_reward * nitrogen_reward +
                  self.weight_nutrient_penalty * nutrient_penalty +
                  self.weight_soil_reward * soil_reward)
        # print(f'Reward: {reward:.2f} (Crop Yield: {crop_yield:.2f}, Nitrogen Reward: {nitrogen_reward:.2f}, Nutrient Penalty: {nutrient_penalty:.2f}, Soil Reward: {soil_reward:.2f})')
        return reward


def train_model(initial_state, num_episodes=10, max_steps_per_episode=100):
    # Create the environment
    env = FertilizerEnvironment(initial_state)

    # Create the DQN agent
    input_size = len(initial_state)
    output_size = len(env.available_fertilizers)
    agent = DQNAgent(input_size, output_size)

    # Define lists to store episode rewards and best results
    episode_rewards = []
    best_rewards = []
    best_fertilizers = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if done:
                break

        # Store episode reward
        episode_rewards.append(episode_reward)

        # Store best result and corresponding fertilizer
        best_rewards.append(env.best_reward)
        best_fertilizers.append(env.best_fertilizer)

        # Print episode statistics
        # print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), episode_rewards, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), best_rewards, label='Best Result')
    plt.xlabel('Episode')
    plt.ylabel('Best Result')
    plt.title('Best Result over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()
    # print('Best Fertilizers:', best_fertilizers)
    # print('Best Rewards:', best_rewards)
    return best_fertilizers, best_rewards





