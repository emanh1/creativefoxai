import numpy as np
import tensorflow as tf
from keras import layers
import envi
import collections
import random

class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

env = envi.ScrcpyGameEnv()

num_actions = np.prod(env.action_space.nvec)  
model = DQN(num_actions)
target_model = DQN(num_actions)
target_model.set_weights(model.get_weights())


optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
loss_function = tf.keras.losses.MeanSquaredError()

ReplayBuffer = collections.deque(maxlen=10000)

def preprocess_observation(observation):
    return observation.astype(np.float32) / 255.0

def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return [np.random.randint(n) for n in env.action_space.nvec]
    else:
        q_values = model.predict(np.expand_dims(state, axis=0))
        return np.unravel_index(np.argmax(q_values[0]), env.action_space.nvec)

def train_step(batch_size, gamma):
    if len(ReplayBuffer) < batch_size:
        return

    mini_batch = random.sample(ReplayBuffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    future_qs = target_model.predict(next_states)
    max_future_qs = np.max(future_qs, axis=1)
    updated_qs = rewards + gamma * max_future_qs * (1 - dones)

    masks = tf.one_hot([np.ravel_multi_index(action, env.action_space.nvec) for action in actions], num_actions)
    with tf.GradientTape() as tape:
        q_values = model(states)
        q_actions = tf.reduce_sum(q_values * masks, axis=1)
        loss = loss_function(updated_qs, q_actions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

epsilon = 1.0  # initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.999
batch_size = 32
num_episodes = 1000
max_steps_per_episode = 1000
gamma = 0.99  
target_update_freq = 10  # update network every 10 episodes

for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_observation(state)
    episode_reward = 0

    for step in range(max_steps_per_episode):
        print(int(step))
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_observation(next_state)
        episode_reward += reward

        ReplayBuffer.append((state, action, reward, next_state, done))

        train_step(batch_size, gamma)

        state = next_state

        if done:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # update target network
    if episode % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}")

env.close()

model.save('model.h5')
