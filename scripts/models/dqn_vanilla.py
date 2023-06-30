import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque 
import sys
sys.path.append('/Users/nhassen/Documents/ProjectQuant/MyRepos/battery-optimisation-with-drl/')

from scripts.models.noise_linear import NoisyLinear
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model architecture (Vanilla)
class QNet(nn.Module):
	""" Policy Model """
	def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128):
		super(QNet, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.dense1 = nn.Linear(state_size, fc1_units)
		# self.dense2 = nn.Linear(fc1_units, fc2_units)
		self.dense3 = nn.Linear(fc1_units, fc2_units)
		self.dense4 = nn.Linear(fc2_units, action_size)

	def forward(self, states):
		""" map state values to action values """
		x = F.relu(self.dense1(states))
		# x = F.relu(self.dense2(x))
		x = F.relu(self.dense3(x))
		return self.dense4(x)

# define model architecture (NN)
class NNQNet(nn.Module):
	"""Policy Model """
	def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=256):
		super(NNQNet, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.dense1 = nn.Linear(state_size, fc1_units)
		# self.dense2 = nn.Linear(fc1_units, fc2_units)
		self.dense3 = NoisyLinear(fc1_units, fc2_units)
		self.dense4 = NoisyLinear(fc2_units, action_size)

	def forward(self, states):
		"""map state values to action values """
		x = F.relu(self.dense1(states))
		# x = F.relu(self.dense2(x))
		x = F.relu(self.dense3(x))
		return self.dense4(x)

	def reset_params(self):
		# self.dense1.reset_noise()
		self.dense3.reset_noise()
		self.dense4.reset_noise()

# replay buffer object
class Replay():
	def __init__(self, action_size, buffer_size, batch_size, seed):
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = self.experiences(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		""" randomly sample experiences from memory """
		experiences = random.sample(self.memory, k=self.batch_size)
      
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().cpu()
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().cpu()
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().cpu()
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().cpu()
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().cpu()

		return (states,actions,rewards,next_states,dones)

	def __len__(self):
		""" get current size of samples in memory """
		return len(self.memory)

# DQN Agent
class DQN_Agent():
	def __init__(self, state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update, qnet_type='vanilla'):
		self.state_size = state_size
		self.action_size = action_size
		self.qnet_type = qnet_type

		# Intialise q-networks
		if self.qnet_type == 'vanilla':
			self.qnet = QNet(state_size, action_size, seed)
			self.qnet_target = QNet(state_size, action_size, seed)
		if self.qnet_type == 'NN':
			self.qnet = NNQNet(state_size, action_size, seed)
			self.qnet_target = NNQNet(state_size, action_size, seed)


		# define optimiser
		self.optimizer = optim.Adam(self.qnet.parameters(), lr=learning_rate)

		# Replay Memory 
		self.memory = Replay(action_size, buffer_size, batch_size, seed)
		self.t_step = 0
		self.soft_update_bool = soft_update        

	def step(self, state, action, reward, next_step, update, batch_size, gamma, tau, done):

		# save expereince in model
		self.memory.add(state, action, reward, next_step, done)

		# learn every 'x' time-steps
		self.t_step = (self.t_step+1) % update

		if self.soft_update_bool == True:
			if self.t_step == 0:
				if len(self.memory) > batch_size:
					experience = self.memory.sample()
					self.learn(experience, gamma, tau)
		else:
			if len(self.memory) > batch_size:            
				experience = self.memory.sample()
				self.learn(experience, gamma, tau)


	def action(self, state, epsilion = 0):
		""" return action for given state given current policy """
		state = torch.from_numpy(state).float().unsqueeze(0).cpu()
		self.qnet.eval()

		with torch.no_grad():
			action_values = self.qnet(state)

		self.qnet.train()    

		# action selection relative to greedy action selection
		if random.random() > epsilion:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))


	def learn(self, experiences, gamma, tau):

		states, actions, rewards, next_states, dones = experiences

		criterion = torch.nn.MSELoss()

		# local model used to train
		self.qnet.train()

		# target model used in eval mode
		self.qnet_target.eval()

		predicted_targets = self.qnet(states).gather(1,actions)

		with torch.no_grad():
			labels_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)

		labels = rewards + (gamma * labels_next*(1-dones))

		loss = criterion(predicted_targets, labels).cpu()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# now update the target next weights
		if self.soft_update_bool == True:      
			self.soft_update(self.qnet, self.qnet_target, tau)
		elif (self.soft_update_bool == False) and self.t_step == 0:
			self.soft_update(self.qnet, self.qnet_target, tau=1)

	def hard_update(self):
		self.qnet_target.load_state_dict(self.qnet.state_dict())

	def soft_update(self, local_model, target_model, tau):

		"""Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target """

		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
















