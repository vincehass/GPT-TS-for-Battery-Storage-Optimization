import sys
sys.path.append('/Users/nhassen/Documents/ProjectQuant/MyRepos/battery-optimisation-with-drl/')
# battery storage optmisation with Reinforcement Learning
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
from pickle import dump


device = "cuda" if torch.cuda.is_available() else "cpu"


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# import model architecture from model scripts directory
sys.path.append('./models') 


# import custom classes:
from scripts.da_electricity_price_model import LSTMCNNModel
from scripts.battery_efficiency import BatteryEfficiency
from scripts.battery_environment import Battery


from scripts.models.dqn_vanilla import DQN_Agent
from scripts.models.dqn_double_dueling import DQN_Agent_double_duel

# seed random number for consistency
seed = 100

# declare environment dictionary params
env_settings = {
	'battery_capacity': 20000,	# rated capacity of battery (kWh)
    'battery_energy': 10000,	# rated power of battery (kW)
    'battery_price': 3,			# battery CAPEX (£/kWh)
    'num_actions': 5,			# splits charge/discharge MWs relative to rated power
    'standby_loss': 0.99,		# standby loss for battery when idle
    'num_episodes': 10,		# number of episodes 1000
    'train': True,				# Boolean to determine whether train or test state
    'scaler_transform_path': '../battery-optimisation-with-drl/data/processed_data/da_price_scaler.pkl',				
    'train_data_path': '../battery-optimisation-with-drl/data/processed_data/train_data_336hr_in_24hr_out_unshuffled.pkl', # Path to trian data
    'test_data_path': '../battery-optimisation-with-drl/data/processed_data/test_data_336hr_in_24hr_out_unshuffled.pkl',	 # Path to test data
    'torch_model': '../battery-optimisation-with-drl/models/da_price_prediction.pt',	 # relevant to current file dir
    'price_track': 'true' # 'true' or 'forecasted'
}
from pickle import dump, load
train_data_path = env_settings['train_data_path']

train_load = open(train_data_path, "rb") 
input_prices = load(train_load)
print(input_prices['X_train'][0])

#check
train_load = open(train_data_path, "rb") 
input_prices = load(train_load)
da_inputs = torch.tensor(input_prices['X_train'][0], dtype=torch.float64)
print(da_inputs)

# DQN models
#dqn_models = ['vanilla', 'double_dueling', 'NN']
# DQN models
#dqn_models = ['double_dueling', 'vanilla', 'double_dueling_NN', 'NN']
dqn_models = ['vanilla', 'double_dueling_NN']
# store profits for each model in dicitonary
dqn_model_profits = {}

# training loop:
for model in dqn_models: # loop for each model type

	# episode + time range parameters
	n_episodes = 10#12000 # max 67925
	time_range = 168

	# e-greedy params

	if model != 'NN' and model != 'double_dueling_NN':
		epsilon = 1.0
		epsilon_end = 0.01
		epsilon_decay = 0.9995
	else:
		epsilon = 0
		epsilon_end = 0
		epsilon_decay = 0

	# further DQN params
	learning_rate = 25e-5 
	buffer_size = int(1e5)
	batch_size = 32 # 64 best
	gamma = 0.99
	# tau = 1e-3
	tau = 0.001 # i.e. 1 'hard' update
	update = 16

	# instaniate environment 
	env = Battery(env_settings)
	state_size = (env.observation_space.shape[0])
	action_size = len(env.action_space)

	print(model)
	# instaniate DQN agent
	if model == "double_dueling" or model == "double_dueling_NN":
		dqn_agent = DQN_Agent_double_duel(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update=True, qnet_type=model)
	else:
		dqn_agent = DQN_Agent(state_size, action_size, learning_rate, buffer_size, gamma, tau, batch_size, seed, soft_update=True, qnet_type=model)        

	total_cumlative_profit = 0
	cumlative_profit = [0]

	# declare arrays to collect info during training
	scores = np.empty((n_episodes)) # list of rewards from each episode
	profits = np.empty((n_episodes))

	for ep in range(n_episodes): # loop through episodes

	    # reset reward and profit vars
		episode_rew = 0
		episode_profit = 0

	    # reset evironment between episodes 
		cur_state = env.reset()

	    # loop through timesteps in each episode
		for step in range(time_range): 

            # action selection
			action = dqn_agent.action(cur_state, epsilon)

	        # env step
			new_state, reward, done, info = env.step(cur_state, action, step)
			print(new_state)
            

	        # agent step
			dqn_agent.step(cur_state, action, reward, new_state, update, batch_size, gamma, tau, done)

			cur_state = new_state 
			episode_rew += reward

			# store episode profit 
			episode_profit += info["ts_cost"]
			cumlative_profit.append(cumlative_profit[-1] + info["ts_cost"])

	        # end episode if 'game over'
			if done:
				break

		scores[ep] = episode_rew 
		profits[ep] = episode_profit
		# epsilon = epsilon - (2/(ep+1)) if epsilon > 0.01 else 0.01
		epsilon = max(epsilon*epsilon_decay, epsilon_end)

		print(f"Episode:{ep}\n Reward:{episode_rew}\n Epsilon:{epsilon}\n Profit: {episode_profit}")
    
	dqn_model_profits[model] = cumlative_profit

	# save model + results from trained DQN model 
	torch.save(dqn_agent.qnet.state_dict(),f'../battery-optimisation-with-drl/scripts/save_ckpt/dqn_{model}.pth')

	with open(f"../battery-optimisation-with-drl/scripts/save_ckpt/rewards_{model}.pkl", "wb") as episode_rewards:
		dump(scores, episode_rewards)

# save timeseries performance df via pickle
with open('../battery-optimisation-with-drl/scripts/save_ckpt/profits.pkl', 'wb') as timeseries_results:
	dump(dqn_model_profits, timeseries_results)

# plot profit comparison
for idx, model in enumerate(dqn_models):
	plt.plot(dqn_model_profits[model], label=f'{model}')

plt.legend(loc="lower right")
plt.show()
