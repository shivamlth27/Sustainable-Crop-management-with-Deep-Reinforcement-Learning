import pandas as pd
import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # Import tqdm for progress tracking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
# print("Device:", device)

# Read the CSV file
df = pd.read_csv(r'C:\Users\Ayush\OneDrive\Desktop\AI-search-algo\ai-project\datasets\Crop_recommendation.csv')
# print(df.head())  # Print the first few rows of the DataFrame to inspect its structure
# print(df.columns)  # Print all column names to verify if 'label' is present

# Scale numerical features to a range between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])


processed_data = pd.concat([pd.DataFrame(scaled_data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']), 
                            df[['label']],  # Retain the original 'label' column
                            pd.get_dummies(df['label'])], axis=1)
# print(processed_data.shape)
# print(processed_data.head())
# print(processed_data['label'].value_counts())
class CustomEnvironment:
    def __init__(self, data):
        self.data = data
        self.num_actions = len(data.columns) - 7
        self.state_shape = data.shape[1] - self.num_actions
        self.max_steps = len(data)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.state = self.data.iloc[self.current_step, :-self.num_actions].values
        return self.state

    def step(self, action):
        self.current_step += 1
        if self.current_step < self.max_steps and self.num_actions < len(self.data.columns):
            next_state = self.data.iloc[self.current_step, :-self.num_actions]
            crop_type = self.data['label'][self.current_step]
            # print(crop_type)
            # print(next_state)
        else:
            next_state = None
            crop_type = None
            return next_state, 0, True
        reward = self.calculate_reward(next_state,crop_type)  # Calculate reward based on the state
        done = False  # Terminate if maximum steps reached
        return next_state, reward, done
    

    def calculate_reward(self, state,crop_type):
            crop_yield = 0
            nitrogen_level = state['N']
            phosphorous_level = state['P']
            potassium_level = state['K']
            temperature = state['temperature']
            humidity = state['humidity']
            ph_level = state['ph']
            rainfall = state['rainfall']
            # Initialize reward
            reward = 0

            # Reward logic for different crop types
            if crop_type == 'maize':
                reward += (50 if nitrogen_level >= 50 else -7) + (80 if phosphorous_level >= 40 else -6) + \
                        (41 if potassium_level >= 40 else -9) + (40 if 20 <= temperature <= 30 else -3) + \
                        (60 if humidity >= 70 else -8) + (20 if 6 <= ph_level <= 7 else -4) + \
                        (100 if rainfall >= 1000 else -5)

            elif crop_type == 'pigeonpeas':
                reward += (90 if phosphorous_level >= 30 else -5) + (70 if potassium_level >= 40 else -8) + \
                        (55 if temperature >= 25 else -6) + (80 if humidity >= 60 else -7) + \
                        (40 if ph_level >= 6.5 else -9) + (30 if rainfall >= 1050 else -4)

            elif crop_type == 'kidneybeans':
                reward += (70 if phosphorous_level >= 30 else -4) + (50 if potassium_level >= 40 else -9) + \
                        (69 if temperature >= 25 else -8) + (40 if humidity >= 60 else -7) + \
                        (90 if ph_level >= 6.5 else -5) + (50 if rainfall >= 1050 else -6)

            elif crop_type == 'cotton':
                reward += (80 if nitrogen_level >= 60 else -7) + (60 if phosphorous_level >= 50 else -8) + \
                        (90 if potassium_level >= 30 else -4) + (70 if 25 <= temperature <= 35 else -6) + \
                        (52 if humidity >= 60 else -9) + (40 if ph_level >= 6 else -5) + \
                        (30 if rainfall >= 1000 else -8)

            elif crop_type == 'coconut':
                reward += (60 if nitrogen_level >= 40 else -9) + (70 if phosphorous_level >= 40 else -5) + \
                        (40 if potassium_level >= 30 else -8) + (90 if 20 <= temperature <= 30 else -6) + \
                        (51 if humidity >= 50 else -7) + (80 if ph_level >= 5.5 else -4) + \
                        (100 if rainfall >= 80 else -3)

            elif crop_type == 'papaya':
                reward += (74 if nitrogen_level >= 80 else -6) + (50 if phosphorous_level >= 30 else -8) + \
                        (60 if potassium_level >= 40 else -9) + (80 if 25 <= temperature <= 35 else -7) + \
                        (70 if humidity >= 80 else -4) + (40 if ph_level >= 5.5 else -5) + \
                        (30 if rainfall >= 1050 else -9)

            elif crop_type == 'orange':
                reward += (70 if nitrogen_level >= 60 else -8) + (80 if phosphorous_level >= 40 else -5) + \
                        (40 if potassium_level >= 30 else -9) + (90 if 105 <= temperature <= 25 else -6) + \
                        (54 if humidity >= 50 else -7) + (60 if ph_level >= 6 else -8) + \
                        (70 if rainfall >= 1000 else -4)

            elif crop_type == 'apple':
                reward += (80 if nitrogen_level >= 70 else -4) + (60 if phosphorous_level >= 50 else -9) + \
                        (90 if potassium_level >= 40 else -7) + (70 if 20 <= temperature <= 30 else -8) + \
                        (50 if humidity >= 60 else -5) + (70 if ph_level >= 6 else -6) + \
                        (40 if rainfall >= 1000 else -9)

            elif crop_type == 'watermelon':
                reward += (90 if nitrogen_level >= 50 else -6) + (70 if phosphorous_level >= 30 else -8) + \
                        (40 if potassium_level >= 30 else -9) + (80 if 20 <= temperature <= 35 else -7) + \
                        (60 if humidity >= 40 else -5) + (50 if ph_level >= 5.5 else -6) + \
                        (30 if rainfall >= 80 else -9)

            elif crop_type == 'grapes':
                reward += (80 if nitrogen_level >= 60 else -7) + (50 if phosphorous_level >= 40 else -8) + \
                        (79 if potassium_level >= 30 else -6) + (90 if 20 <= temperature <= 35 else -9) + \
                        (40 if humidity >= 50 else -7) + (60 if ph_level >= 6 else -8) + \
                        (70 if rainfall >= 80 else -4)

            elif crop_type == 'pomegranate':
                reward += (49 if nitrogen_level >= 50 else -8) + (40 if phosphorous_level >= 40 else -6) + \
                        (56 if potassium_level >= 30 else -9) + (60 if 105 <= temperature <= 30 else -7) + \
                        (73 if humidity >= 50 else -5) + (80 if ph_level >= 6 else -6) + \
                        (96 if rainfall >= 1000 else -8)

            elif crop_type == 'lentil':
                reward += (82 if nitrogen_level >= 60 else -9) + (80 if phosphorous_level >= 40 else -7) + \
                        (89 if potassium_level >= 30 else -5) + (60 if 20 <= temperature <= 35 else -6) + \
                        (23 if humidity >= 50 else -7) + (80 if ph_level >= 6 else -8) + \
                        (19 if rainfall >= 1000 else -9)


            return abs(reward)

# Create an instance of your custom environment
env = CustomEnvironment(processed_data)

# Define a more complex deep RL model using PyTorch
class CustomModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize model and optimizer
input_shape = env.state_shape
num_actions = env.num_actions
print(input_shape)
print(num_actions)
model = CustomModel(input_shape, num_actions).to(device)  # Move model to device
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define other training parameters
num_episodes = 10
batch_size = 32
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
gamma = 0.99
buffer_size = 10000
replay_buffer = []

# Training loop with tqdm progress bar
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    # Use tqdm for progress tracking
    for _ in tqdm(range(len(env.data))):
        # Epsilon-geedy policy for exploration
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
            action = torch.argmax(q_values).item()
        
        next_state, reward, done = env.step(action)
        episode_reward += reward
        
        # Store experience in replay buffer
        if next_state is not None:
            state = state.astype(np.float32)  # Convert state to float32
            action = int(action)  # Convert action to int64
            reward = float(reward)  # Convert reward to float32
            next_state = next_state.astype(np.float32)  # Convert next_state to float32
            done = bool(done)  # Convert done to bool
            replay_buffer.append((state, action, reward, next_state, done))
        
        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)  # Remove oldest experience if buffer size exceeded
        
        # Sample mini-btch from replay buffer if buffer size is larger than batch size
        if len(replay_buffer) >= batch_size:
            batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[idx] for idx in batch_indices]

            # Check if any batch item has None in next_state
            batch = [(s.astype(np.float32), int(a), float(r), ns.astype(np.float32), bool(d)) for s, a, r, ns, d in batch if ns is not None]

            states_batch = torch.FloatTensor(np.array([item[0] for item in batch])).to(device)
            action_batch = torch.LongTensor(np.array([item[1] for item in batch])).to(device)
            reward_batch = torch.FloatTensor(np.array([item[2] for item in batch])).to(device)
            next_states_batch = torch.FloatTensor(np.array([item[3] for item in batch])).to(device)
            done_batch = torch.BoolTensor(np.array([item[4] for item in batch])).to(device)

            # Calculate target Q-vlues using Bellman equation
            with torch.no_grad():
                target_q_values = reward_batch + torch.logical_not(done_batch) * gamma * torch.max(
                    model(next_states_batch), dim=1
                )[0]
            # Compute loss and perform gradient descent
            optimizer.zero_grad()
            predicted_q_values = model(states_batch)
            predicted_q_values_actions = torch.sum(
                predicted_q_values * torch.nn.functional.one_hot(action_batch, num_actions), dim=1
            )
            loss = torch.nn.functional.mse_loss(target_q_values, predicted_q_values_actions)
            loss.backward()
            optimizer.step()
        
        state = next_state
        
        if done:
            break

    # Decay epsilon for epsilon-geedy policy
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode: {episode + 1}, Episode Reward: {episode_reward}")
