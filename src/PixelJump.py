# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
	from malmo import MalmoPython
except:
	import MalmoPython

import get_xml as GM
import sys
import time
import json
import random
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

class PixelJump(gym.Env):

	def __init__(self, env_config):  
		# Static Parameters

		# ===================== XML Hyper Parameters ============================

		# Map Difficulty Level
		# 1. Complete Platform with goal block always centered
		# 2. Complete Platform with goal block randomly at z-axis
		# 3. Complete Platform with goal block randomly placed
		# 4. Incomplete Platform with goal block randomly placed or might not have one
		self.difficulty = 3

		# Map size
		self.size = 300
		self.floor = 3

		# Gap size between each platfrom
		self.gap_min = 2
		self.gap_max = 4             

		# Platform definition
		self.block_density = 0.4   # probability of each block
		self.block_size = 1      # 3x3 platform
		self.goal_block_density = 0.9

		# Direction 
		self.direction_freq = 0.3 # change frequency density

		# Jumping range displacement
		self.velocity_min =  7.81 # range = 4m (Minimum possible distance)
		self.velocity_max = 11.72 # range = 9m (Maximum possible distance)

		# Platform block types
		self.block_types = ['iron_block', 'emerald_block', 'gold_block', 'lapis_block', 'diamond_block', 'redstone_block', 'purpur_block']

		# reward system
		self.penalty = -10
		self.goal_reward = 100

		# ===================== PixelJump Parameters ====================

		# graph/plot parameter control
		self.max_episode_steps = 50
		self.log_frequency_eps = 500
		self.log_frequency_stp = 1000

		# observation space parameters
		self.obs_size_x = 5
		self.obs_size_z = 10
		self.obs = None


		self.steps = 0
		self.episode = 0
		self.episode_step = 0
		self.episode_score = 0

		self.step_scores = []        # store the number of scores in a step
		self.step_relative_diff = [] # store number of relative diff in a step

		self.episode_scores = [] # store numbers of scores in an episode
		self.episode_steps = []  # store numbers of steps in an episode


		# ===================== Step Parameters ===========================
		self.XPos = 1.5
		self.YPos = 3
		self.ZPos = 1.5

		self.velocity = 0
		self.degree = 0
		self.relative_pos = 0
		self.relative_pos_x = 0
		self.relative_pos_z = 0


		# Rllib Parameters
		self.action_space = Box(0, 1, shape=(2,), dtype=np.float32) # used to determine its degree and the chosne velocity
		self.observation_space = Box(0, 1, shape=(np.prod([2, self.obs_size_z, self.obs_size_x]), ), dtype=np.int32)

		# Malmo Parameters
		self.agent_host = MalmoPython.AgentHost()
		try:
			self.agent_host.parse( sys.argv )
		except RuntimeError as e:
			print('ERROR:', e)
			print(self.agent_host.getUsage())
			exit(1)


	def reset(self):
		"""
		Resets the environment for the next episode.
		Returns
		observation: <np.array> flattened initial obseravtion
		"""
		# Reset Malmo
		world_state = self.init_malmo()

		# Reset Variables
		self.episode += 1
		self.episode_steps.append(self.episode_step)
		self.episode_scores.append(self.episode_score)


		if len(self.episode_steps) > 1 and self.episode > 0:     
			print("Episode {} return: {}".format(self.episode, self.episode_score/self.episode_step))
			print("Avg return: {}\n".format(sum(self.episode_scores)/(self.episode)))
			print("========================================================")    

		self.episode_step = 0
		self.episode_score = 0

		self.XPos = 1.5
		self.YPos = 3
		self.ZPos = 1.5


		# Log
		if self.steps > 10000:
			self.log_frequency_eps /= 2
		elif self.steps > 5000:
			self.log_frequency_eps /= 5


		if self.episode > self.log_frequency_eps and \
			self.episode % self.log_frequency_eps == 0:
			self.log_episode()

		if self.steps > self.log_frequency_stp and \
			self.steps % self.log_frequency_stp == 0:
			self.log_steps()

		# Get Observation
		self.obs = self.get_observation(world_state)

		return self.obs.flatten()


	def movement (self, v, x, y, z, degree):
		ax = 0
		az = 0 
		ay = -9.8  
		t = 0.08
		d = np.radians(70) 
		degree = -1*degree+90 # Adjust degree in minecraft to real degree in trigonometry.
							  # For example, 0 degree in minecraft is viewed as 90 degree in a xz plane, 
							  # a simple solution is to add 90 degree to the original degree,
							  # Caution, in mincraft positive degree resssults from clockwise rotation,
							  # negative degree results form counter-clockwise rotation, which is the inverse of general measurement.
		M = []

		vx = v * np.cos(d) * np.cos(np.radians(degree)) # cos(degree) give the ratio of x after transformation
		vz = v * np.cos(d) * np.sin(np.radians(degree)) # sin(degree) give the ratio of z after transformation
		vy = v * np.sin(d)

		while True:
			x = x + vx*t
			z = z + vz*t
			y = y + vy*t + 0.5*ay*(t**2)

			vx = vx + ax*t
			vz = vz + az*t
			vy = vy + ay*t

			if y < self.floor:
				break

		M.append([x,y,z])
		return M


	def perform_jump(self, movementPath):
		path = []
		for a in movementPath:
			x,y,z = a[0],a[1],a[2]
			path.append("tp {} {} {}".format(round(x,4),round(y,4), round(z,4)))

		self.XPos = x
		self.YPos = y
		self.ZPos = z
		return path


	def step(self, action):
		"""
		Take an action in the environment and return the results.
		Args
		action: <int> index of the action to take
		Returns
		observation: <np.array> flattened array of obseravtion
		reward: <int> reward from taking action
		done: <bool> indicates terminal state
		info: <dict> dictionary of extra information
		""" 

		# Perform a jump based on user picked actions
		self.velocity = self.velocity_min + ((self.velocity_max-self.velocity_min) * action[0])

		left_theta = np.degrees(np.arctan((3-self.XPos) / (self.gap_min+1)))
		right_theta = np.degrees(np.arctan(self.XPos / (self.gap_min+1)))
		theta = left_theta + right_theta

		if self.XPos >= 1.5:
			self.degree = -left_theta + theta * action[1]
		elif self.XPos < 1.5:
			self.degree = -right_theta + theta * action[1]

		movements = self.movement(self.velocity, self.XPos, self.YPos, self.ZPos, self.degree)#self.degree)
		commands = self.perform_jump(movements)

		# for c in commands:
		#     time.sleep(0.05)
		#     self.agent_host.sendCommand(c)

		self.agent_host.sendCommand(commands[-1])
		time.sleep(2)

		self.steps += 1
		self.episode_step += 1

		# Get Done
		done = False
		if self.episode_step >= self.max_episode_steps:
			done = True
			time.sleep(2)  

		# Get Observation
		XRel = self.relative_pos_x
		ZRel = self.relative_pos_z
		world_state = self.agent_host.getWorldState()
		for error in world_state.errors:
			print("Error:", error.text)
		self.obs = self.get_observation(world_state) 


		# Get Reward
		score = 0
		self.relative_pos = np.sqrt((self.XPos-XRel)**2 + (self.ZPos-ZRel)**2)
		self.step_relative_diff.append(self.relative_pos)

		for r in world_state.rewards:
			score = r.getValue()
			if score == self.goal_reward:
				self.relative_pos = 0
			elif score <= self.penalty:
				score = self.penalty
				done = True
			elif score > self.goal_reward: # step on the border of multiple blocks
				score = self.goal_reward-10

		score -= (self.relative_pos*10)

		self.episode_score += score
		self.step_scores.append(score)

		# ============================ Output ============================================
		print("\nStep: {}".format(self.steps))
		print("Step Score: {}\n".format(score))

		print("Velocity: {}".format(self.velocity))
		print("Degree:   {}".format(self.degree))
		# print("Theta:    {}".format(theta))
		# print("Glass   Position: ({},{})".format(XRel, ZRel))
		# print("Current Position: ({},{})".format(round(self.XPos,2), round(self.ZPos,2)))
		print("Relative distance: {}\n".format(self.relative_pos))

		return self.obs.flatten(), score, done, dict()


	def get_observation(self, world_state):
		"""
		Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
		The agent is in the center square facing up.
		Args
		world_state: <object> current agent world state
		Returns
		observation: <np.array>
		"""
		obs = np.zeros((2,self.obs_size_z, self.obs_size_x))

		while world_state.is_mission_running:
			time.sleep(0.1)
			world_state = self.agent_host.getWorldState()
			if len(world_state.errors) > 0:
				raise AssertionError('Could not load grid.')

			if world_state.number_of_observations_since_last_state > 0:
				# First we get the json from the observation API
				msg = world_state.observations[-1].text
				observations = json.loads(msg)

				# Get observation
				grid = observations['self.floorAll']

				row = 0
				blocks = 0
				platform_row = 0
				firstBlock = True

				self.relative_pos = 0
				self.relative_pos_x = 0
				self.relative_pos_z = 0

				grid_blocks = []
				grid_glass = []
				for x in grid:        
					if blocks%5 == 0:
						row += 1
					if row > 2 and (row-platform_row <= 3 or platform_row == 0):
						if x in self.block_types:
							grid_glass.append(0)
							grid_blocks.append(1)
							if firstBlock:
								self.relative_pos_x = self.XPos + (blocks%5-2)
								self.relative_pos_z =  self.ZPos + row
								firstBlock = False
							if platform_row == 0:
								platform_row = row
						elif x == "glass":
							grid_blocks.append(1)
							grid_glass.append(1)
							self.relative_pos_x = self.XPos + (blocks%5-2)
							self.relative_pos_z = self.ZPos + row
						else:
							grid_blocks.append(0)
							grid_glass.append(0)
					else:
						grid_blocks.append(0)
						grid_glass.append(0)
					blocks += 1

				obs = np.reshape(grid_blocks+grid_glass, (2, self.obs_size_z, self.obs_size_x))
				break   

		return obs

	def init_malmo(self):
		"""
		Initialize new malmo mission.
		"""
		my_mission = MalmoPython.MissionSpec(GM.get_mission_xml(self), True)
		my_mission_record = MalmoPython.MissionRecordSpec()
		my_mission.requestVideo(800, 500)
		my_mission.setViewpoint(1)

		max_retries = 3
		my_clients = MalmoPython.ClientPool()
		my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

		for retry in range(max_retries):
			try:
				self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'PixelJump' )
				break
			except RuntimeError as e:
				if retry == max_retries - 1:
					print("Error starting mission:", e)
					exit(1)
				else:
					time.sleep(2)

		world_state = self.agent_host.getWorldState()
		while not world_state.has_mission_begun:
			time.sleep(0.1)
			world_state = self.agent_host.getWorldState()
			for error in world_state.errors:
				print("\nError:", error.text)

		return world_state



	def log_episode(self):

		with open('episodes.txt', 'w') as f:
			i = 1
			for step, value in zip(self.episode_steps, self.episode_scores):
				f.write("{}\t{}\t{}\n".format(i, step, value)) 
				i += 1

	def log_steps(self):

		with open('steps.txt', 'w') as f:
			i = 1 
			for value, diff in zip(self.step_scores, self.step_relative_diff):
				f.write("{}\t{}\t{}\n".format(i, value, diff)) 
				i += 1



if __name__ == '__main__':
	ray.init()
	trainer = ppo.PPOTrainer(env=PixelJump, config={
		'env_config': {},           # No environment parameters to configure
		'framework': 'torch',       # Use pyotrch instead of tensorflow
		'num_gpus': 0,              # We aren't using GPUs
		'num_workers': 0            # We aren't using parallelism
	})

	while True:
		print(trainer.train())
