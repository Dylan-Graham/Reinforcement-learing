import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1')

states = env.observation_space.shape[0]
actions = env.action_space.n


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=24):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DQN(states, actions)

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

agent = DQNAgent(
    model=model,
    memory=memory,
    policy=policy,
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check that the model returns a tensor of the correct shape
test_input = torch.randn((1, states))
test_output = model(test_input)
assert test_output.shape == (1, actions)

# Mean absolute error = mae
agent.compile(optimizer=optimizer, metrics=['mae'])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history['episode_reward']))

env.close()
