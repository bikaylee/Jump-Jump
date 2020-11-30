# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

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



# Hyperparameters
FLOOR = 3

# Map size
SIZEx = 300
SIZEz = 300

# Gap size between each platfrom
GAP_MIN = 2
GAP_MAX = 4             

# Platform definition
BLOCK_DENSITY = 0.4   # probability of each block
BLOCK_SIZE = 1      # 3x3 platform

# Direction 
DIRECTION_FREQ = 0.3 # change frequency density

# Jumping range displacement
VELOCITY_MIN = 6.77  # range = 3m (Minimum possible distance)
VELOCITY_MAX = 11.72 # range = 9m (Maximum possible distance)

# Platform block types
BLOCK_TYPES = ['iron_block', 'emerald_block', 'gold_block', 'lapis_block', 'diamond_block', 'redstone_block', 'purpur_block']

class PixelJump(gym.Env):

    def GetMissionXML():

        # Starting platform
        xml = "<DrawCuboid x1='0' x2='2' y1='2' y2='2' z1='0' z2='2' type='stone'/>"

        z = 2 + random.randint(GAP_MIN, GAP_MAX) + 2 # first platform 
        x = 1

        platform_index = 1
        max_platform = 50
        
        while platform_index < max_platform:
            block = random.choice(BLOCK_TYPES)

            arr = [1, 0, -1]
            for i in range(3):
                for j in range(3):
                    if random.random() < BLOCK_DENSITY:
                        xml += "<DrawBlock x='{}' y='2' z='{}' type='{}'/>".format(x+arr[j], z+arr[i], block)

            # Center block
            xml += "<DrawBlock x='{}' y='2' z='{}' type='glass'/>".format(x, z)


            if random.random() < DIRECTION_FREQ:
                x += random.randint(GAP_MIN, GAP_MAX) + 3
            else:
                z += random.randint(GAP_MIN, GAP_MAX) + 3
            platform_index += 1

             

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Jump Jump Jump!</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>true</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' +\
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-50, SIZEx, -50, SIZEz) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-50, SIZEx, -50, SIZEz) + \
                                xml +\
                                '''
                            </DrawingDecorator>
                            <ServerQuitFromTimeUp timeLimitMs="500000"/>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Creative">
                        <Name>James Bond</Name>
                        <AgentStart>
                            <Placement x="1.5" y="3" z="1.5" pitch="0"/>
                        </AgentStart>
                        <AgentHandlers>
                            <RewardForTouchingBlockType>
                                <Block type='glass' reward='100' />
                                <Block type='iron_block emerald_block gold_block lapis_block diamond_block redstone_block purpur_block' reward='10' />
                                <Block type='lava' reward='-10' />
                            </RewardForTouchingBlockType>
                            <AbsoluteMovementCommands/>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''



    def movement (self, v, x ,y):
        ax = 0 
        ay = -9.8  
        t = 0.08
        d = np.radians(70) 

        M = []

        vx = v * np.cos(d)
        vy = v * np.sin(d)

        while True:
            x = x + vx*t
            y = y + vy*t + 0.5*ay*(t**2)
            vx = vx + ax*t
            vy = vy + ay*t

            if y < FLOOR:
                break
            M.append([x,y])
        return M


#     def perform_jump(self, movementPath, direction):
#         path = []
#         for a in movementPath:
#             y = a[1]
#             if direction: # 1 = z-positive direction
#                 z = a[0]
#                 path.append("tp 0.5 {} {}".format(round(y,4),round(z,4)))
#             else: # x-positive direction
#                 x = a[0]
#                 path.append("tp {} {} 0.5".format(round(x,4),round(y,4)))
#         return path


#         x = 0.5
#         action_list = []
#         for i in range(10):
#             velocity = random.uniform(VELOCITY_MIN, VELOCITY_MAX)
#             movement_path = movement(velocity, x, 3)
#             x = movement_path[-1][0]
#             action_list.append(perform_jump(movement_path))
        
#         for a in action_list:
#             time.sleep(0.5)
#             print()
#             for index in range(len(a)):
#                 agent_host.sendCommand(a[index])
#                 time.sleep(0.02)
#                 print(a[index])

    def perform_jump(self, movementPath, moving_axis, XPos, ZPos):
        path = []
        if(moving_axis == 0):
            for a in movementPath:
                z = a[0]
                y = a[1]
                path.append("tp {} {} {}".format(XPos,round(y,4), round(z,4)))
        else:
            for a in movementPath:
                x = a[0]
                y = a[1]
                path.append("tp {} {} {}".format(round(x,4),round(y,4), ZPos))
        return path

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.axis_dict = {
            0: 1, 
            1: 0
        }
        self.moving_axis = 0  # 0: z, 1: x


        # Rllib Parameters
        # self.action_space = Box()
        self.degree_threshold = (11.72 - 6.77)/3 #threshold for turn degree determination
        self.action_space = Box(6.77, 11.72, shape = (2,), dtype = np.float32) # action[0] is used to determine its degree, action[1] is the chosne velocity
        self.observation_space = Box(0, 1, 2, shape=(np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)


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
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()

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

        # Get Cuurent Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        XPos = obs['XPos']
        YPos = obs['YPos']
        ZPos = obs['ZPos']

        if (action[0] <= self.degree_threshold):
            self.agent_host.sendCommand("turn -90")
            self.moving_axis = self.axis_dict.get(self.moving_axis)
        elif (action[0] > self.degree_threshold * 2):
            self.agent_host.sendCommand("turn 90")
            self.moving_axis = self.axis_dict.get(self.moving_axis)

        if (self.moving_axis == 0):
            movements = self.perform_jump(self.movement(action[1], ZPos, YPos), self.moving_axis, XPos, ZPos)
        elif (self.moving_axis == 1):
            movements = self.perform_jump(self.movement(action[1], XPos, YPos), self.moving_axis, XPos, ZPos)

        for m in movements:
            self.agent_host.sendCommand(m)
            time.sleep(0.5)
        self.episode_step += 1


        # Get Done
        done = False
        if self.episode_step >= self.max_episode_steps):
            done = True
            time.sleep(2)  

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
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

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = np.zeros((2, self.obs_size, self.obs_size))

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
                grid = observations['floorAll']
                grid_binary = [1 if x == 'diamond_ore' or x == 'lava' else 0 for x in grid]
                obs = np.reshape(grid_binary, (2, self.obs_size, self.obs_size))

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                
                break

        return obs

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

