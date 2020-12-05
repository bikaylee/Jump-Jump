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
        # Map size
        self.size = 300
        self.floor = 3
        self.difficulty = 3
        
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

        self.obs_size_x = 5
        self.obs_size_z = 10
        self.max_episode_steps = 50
        self.log_frequency = 100

        self.XPos = 1.5
        self.YPos = 3
        self.ZPos = 1.5

        self.velocity = 0
        self.degree = 0
        self.relative_pos = -1
        self.relative_pos_x = 0
        self.relative_pos_z = 0
        self.episode_distance = 0
        self.episode_distances = []
        self.relative_differences = []

        self.penalty = -10
        self.goal_reward = 100

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

        # PixelJump Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.current_step = 0
        self.episode = 0
        self.episode_score = []
        self.cur_step = 0

    def reset(self):
        """
        Resets the environment for the next episode.
        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        self.current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(self.current_step + self.episode_step)
        self.episode_distances.append(self.episode_distance)

        l = len(self.returns)
        e = len(self.episode_score)
        if l > 1 and e > 0:     
            print("\nEpisode return: {}".format(sum(self.episode_score)/e))
            print("Avg return: {}".format(sum(self.returns)/(l-1)))
            print("\n========================================================")    

        self.episode += 1
        self.episode_return = 0
        self.episode_step = 0
        self.episode_score = []
        self.episode_distance = 0

        self.XPos = 1.5
        self.YPos = 3
        self.ZPos = 1.5
        self.relative_pos = -1
        self.relative_pos_x = 0
        self.relative_pos_z = 0


        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()

    
    def movement (self, v, x, y, z, degree):
        ax = 0
        az = 0 
        ay = -9.8  
        t = 0.08
        d = np.radians(70) 

        M = []

        vx = v * np.tan(np.radians(degree))
        vz = v * np.cos(d)
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
        step_pos_x = self.XPos
        step_pos_z = self.ZPos

        velocity_diff = self.velocity_max-self.velocity_min 
        self.velocity = self.velocity_min + (velocity_diff * action[0])


        left_theta = np.degrees(np.arctan((3-self.XPos) / (self.gap_min+1)))
        right_theta = np.degrees(np.arctan(self.XPos / (self.gap_min+1)))
        theta = left_theta + right_theta

        if self.XPos >= 1.5:
            self.degree = -left_theta + theta * action[1]
        elif self.XPos < 1.5:
            self.degree = -right_theta + theta * action[1]


        movements = self.movement(self.velocity, self.XPos, self.YPos, self.ZPos, self.degree)
        commands = self.perform_jump(movements)

        # for c in commands:
        #     time.sleep(0.05)
        #     self.agent_host.sendCommand(c)

        self.agent_host.sendCommand(commands[-1])
        time.sleep(2)
        self.episode_step += 1
        self.cur_step += 1

        # Get Done
        done = False
        if self.episode_step >= self.max_episode_steps:
            done = True
            time.sleep(2)  

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Reward
        step_pos_z = self.ZPos - step_pos_z
        step_pos_x = self.XPos - step_pos_x
        score = 0
        self.relative_pos = np.sqrt( (step_pos_x-self.relative_pos_x)**2 + (step_pos_z-self.relative_pos_z)**2)
        self.relative_differences.append(self.relative_pos)
        for r in world_state.rewards:
            score = r.getValue()
            if score == self.goal_reward:
                self.relative_pos = 0
                self.episode_distance += 1
            elif score == self.penalty:
                done = True
            else: # if score != self.penalty and score != self.goal_reward:
                score = round((1-(self.relative_pos/15)) * (self.goal_reward-30), 4)
                self.episode_distance += 1
        self.episode_score.append(score)
        self.episode_return += score

        print("\nStep: {}".format(self.cur_step))
        print("Step Score: {}".format(score))

        print("Velocity: {}".format(self.velocity))
        print("Degree:   {}".format(self.degree))
        print("Theta:    {}".format(theta))

        print("Current Position: ({},{})".format(round(step_pos_z,2), round(step_pos_x,2)))
        print("Glass   Position: ({},{})".format(self.relative_pos_z, self.relative_pos_x))
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

                num_zero = 0

                grid_blocks = []
                grid_glass = []
                for x in grid:        
                    blocks += 1
                    if blocks%5 == 0:
                        row += 1
                    if row > 2 and (row-platform_row <= 3 or platform_row == 0):
                        if x in self.block_types:
                            grid_glass.append(0)
                            grid_blocks.append(1)
                            if platform_row == 0:
                                platform_row = row
                        elif x == "glass":
                            grid_blocks.append(1)
                            grid_glass.append(1)
                            self.relative_pos_x = np.abs(blocks%5/2)
                            self.relative_pos_z = row+1
                        else:
                            grid_blocks.append(0)
                            grid_glass.append(0)
                            num_zero += 1
                    else:
                        grid_blocks.append(0)
                        grid_glass.append(0)
                        num_zero += 1

                obs = np.reshape(grid_blocks+grid_glass, (2, self.obs_size_z, self.obs_size_x))

                # if num_zero < 50:
                #     print()
                #     print(obs)

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

    def log_returns(self):
        """
        Log the current returns as a graph and text file
        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('Pixel Jump')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')
        
        plt.clf()
        plt.plot(range(self.episode), self.episode_distances)
        plt.title('Pixel Jump Ep Distance')
        plt.ylabel('Distances')
        plt.xlabel('Episodes')
        plt.savefig('distance_returns.png')

        plt.clf()
        plt.plot(range(1,self.steps[-1]+1), self.relative_differences)
        plt.title('Pixel Jump Relative Differences')
        plt.ylabel('Difference')
        plt.xlabel('Step')
        plt.savefig('differences_returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value)) 


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
