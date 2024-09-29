import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2  
import envi
import collections
import random
import gc

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(device)
    print("GPU Name:", gpu_name)
else:
    print("No GPU available.")
class DuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust based on the input size after convolutions
        self.fc_value = nn.Linear(512, 1)
        self.fc_adv = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))

        value = self.fc_value(x)
        adv = self.fc_adv(x)

        q_values = value + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

# Preprocess observation by resizing to a smaller resolution
def preprocess_observation(observation):
    observation = cv2.resize(observation, (84, 84))  # Resize to 84x84 to save memory
    observation = observation.transpose((2, 0, 1))  # Convert to channel-first format (PyTorch)
    return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) / 255.0  # Normalize

env = envi.ScrcpyGameEnv()

num_actions = np.prod(env.action_space.nvec)
model = DuelingDQN(num_actions).to(device)
target_model = DuelingDQN(num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.00025)
loss_fn = nn.MSELoss()

# Reduce replay buffer size to 5000 to save memory
ReplayBuffer = collections.deque(maxlen=5000)

def choose_action(state, epsilon):
    if random.random() <= epsilon:
        return [random.randint(0, n - 1) for n in env.action_space.nvec]
    else:
        with torch.no_grad():
            q_values = model(state)
            return np.unravel_index(q_values.argmax().item(), env.action_space.nvec)

def train_step(batch_size, gamma):
    if len(ReplayBuffer) < batch_size:
        return

    mini_batch = random.sample(ReplayBuffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    states = torch.cat(states).to(device)
    actions = torch.tensor([np.ravel_multi_index(action, env.action_space.nvec) for action in actions], dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = model(states)
    q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q_values = target_model(next_states)
        max_next_q_values = next_q_values.max(1)[0]

    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    loss = loss_fn(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Hyperparameters
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995  # Less aggressive decay
batch_size = 16  # Reduced batch size to save memory
num_episodes = 1000
max_steps_per_episode = 1000
gamma = 0.99
target_update_freq = 10  # Update network every 10 episodes

for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_observation(state)
    episode_reward = 0

    for step in range(max_steps_per_episode):
        print("Step", int(step))
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_observation(next_state)
        episode_reward += reward

        ReplayBuffer.append((state, action, reward, next_state, done))

        train_step(batch_size, gamma)

        state = next_state
        print("Episode Reward", episode_reward)
        if done:
            break

    # Update epsilon to reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Update target network every target_update_freq episodes
    if episode % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}")

    # Explicitly call garbage collector to free memory
    gc.collect()

env.close()

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
