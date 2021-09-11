from arguments import get_args
from Dagger import DaggerAgent, ExampleAgent, MyDaggerAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def plot(record):
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(record['steps'], record['mean'],
		      color='blue', label='reward')
	ax.fill_between(record['steps'], record['min'], record['max'],
	                color='blue', alpha=0.2)
	ax.set_xlabel('number of steps')
	ax.set_ylabel('Average score per episode')
	ax1 = ax.twinx()
	ax1.plot(record['steps'], record['query'],
	         color='red', label='query')
	ax1.set_ylabel('queries')
	reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
	query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
	patch_set = [reward_patch, query_patch]
	ax.legend(handles=patch_set)
	fig.savefig('performance.png')


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
	def __init__(self, env_name, num_stacks):
		self.env = gym.make(env_name)
		# num_stacks: the agent acts every num_stacks frames
		# it could be any positive integer
		self.num_stacks = num_stacks
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def step(self, action):
		reward_sum = 0
		for stack in range(self.num_stacks):
			obs_next, reward, done, info = self.env.step(action)
			reward_sum += reward
			if done:
				self.env.reset()
				return obs_next, reward_sum, done, info
		return obs_next, reward_sum, done, info

	def reset(self):
		return self.env.reset()


def main():
	# load hyper parameters
	args = get_args()
	num_updates = int(args.num_frames // args.num_steps)
	start = time.time()
	record = {'steps': [0],
	          'max': [0],
	          'mean': [0],
	          'min': [0],
	          'query': [0]}
	# query_cnt counts queries to the expert
	query_cnt = 0

	human2Label = {0:0, 2:5, 4:4, 5:1, 6:3, 7:7, 8:2, 9:6}
	label2Actoin = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:11, 7:12}
	# environment initial
	envs = Env(args.env_name, args.num_stacks)
	# action_shape is 18
	# Most of the 18 actions are useless, find important actions
	# in the tips of the homework introduction document
	action_shape = envs.action_space.n
	# observation_shape is the shape of the observation
	# here is (210,160,3)=(height, weight, channels)
	observation_shape = envs.observation_space.shape

	# agent initial
	# you should finish your agent with DaggerAgent
	# e.g. agent = MyDaggerAgent()
	agent = MyDaggerAgent()

	# You can play this game yourself for fun
	if args.play_game:
		obs = envs.reset()
		while True:
			envs.env.render()
			im = Image.fromarray(obs)
			im.save('./imgs/' + str('screen') + '.jpeg')
			action = int(input('input action'))
			while action not in humanActMap.keys():
				action = int(input('re-input action'))
			action = humanActMap(action)
			obs_next, reward, done, _ = envs.step(action)
			obs = obs_next
			if done:
				obs = envs.reset()
                

	data_set = {'data': [], 'label': []}
	# start train your agent
	for i in range(num_updates):
		# an example of interacting with the environment
		# we init the environment and receive the initial observation
		obs = envs.reset()
		# we get a trajectory with the length of args.num_steps
		for step in range(args.num_steps):
			if i == 0:
				envs.env.render()
				human = int(input('input human action'))
				while human not in human2Label.keys():
					human = int(input('re-input human action'))
				label = human2Label(human)
				action = label2Action(label)
				obs_next, reward, done, _ = envs.step(action)
				obs = obs_next
				if done:
					obs = envs.reset()
					
				now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
				im = Image.fromarray(obs)
				im.save('./imgs/' + str(label) + '/' + now + '.jpeg')
			else:
				# we choose a special action according to our model
				label = agent.select_action(obs)
				action = label2Action(label)
			
			# interact with the environment
			# we input the action to the environments and it returns some information
			# obs_next: the next observation after we do the action
			# reward: (float) the reward achieved by the action
			# done: (boolean)  whether it’s time to reset the environment again.
			#           done being True indicates the episode has terminated.
			obs_next, reward, done, _ = envs.step(action)
			# we view the new observation as current observation
			obs = obs_next
			# if the episode has terminated, we need to reset the environment.
			if done:
				envs.reset()

			data_set['data'].append(obs)
			data_set['label'].append(label)

		# design how to train your model with labeled data
		agent.update(data_set['data'], data_set['label'])

		if (i + 1) % args.log_interval == 0:
			total_num_steps = (i + 1) * args.num_steps
			obs = envs.reset()
			reward_episode_set = []
			reward_episode = 0
			# evaluate your model by testing in the environment
			for step in range(args.test_steps):
				action = agent.select_action(obs)
				# you can render to get visual results
				# envs.render()
				obs_next, reward, done, _ = envs.step(action)
				reward_episode += reward
				obs = obs_next
				if done:
					reward_episode_set.append(reward_episode)
					reward_episode = 0
					envs.reset()

			end = time.time()
			print(
				"TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
					.format(
					time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
					i, total_num_steps,
					int(total_num_steps / (end - start)),
					query_cnt,
					np.mean(reward_episode_set),
					np.min(reward_episode_set),
					np.max(reward_episode_set)
				))
			record['steps'].append(total_num_steps)
			record['mean'].append(np.mean(reward_episode_set))
			record['max'].append(np.max(reward_episode_set))
			record['min'].append(np.min(reward_episode_set))
			record['query'].append(query_cnt)
			plot(record)


if __name__ == "__main__":
	main()

