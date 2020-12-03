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

class PixelJump(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters

        self.floor = 3

        # Map size
        self.size = 300
        
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
        self.velocity_min = 7.81  # range = 4m (Minimum possible distance)
        self.velocity_max = 11.72 # range = 9m (Maximum possible distance)

        # Platform block types
        self.block_types = ['iron_block', 'emerald_block', 'gold_block', 'lapis_block', 'diamond_block', 'redstone_block', 'purpur_block']

        self.obs_size_x = 5
        self.obs_size_z = 9
        self.max_episode_steps = 100
        self.log_frequency = 10

        self.XPos = 1.5
        self.YPos = 3
        self.ZPos = 1.5

        self.velocity = 0
        self.degree = 0
        self.relative_pos = 0
        self.midpoint = [0,0]
        # self.total_steps = 0
        
        # Rllib Parameters
        self.action_space = Box(0, 1, shape=(2,), dtype=np.float32) # used to determine its degree and the chosne velocity
        self.observation_space = Box(0, 2, shape=(np.prod([self.obs_size_x, self.obs_size_z]), ), dtype=np.int32)

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

        l = len(self.returns)
        if (l > 1):         
              
            print("Episode {} return: {}".format(len(self.returns)-1, self.episode_return))
            print("Avg Ep return: {}".format(sum(self.returns)/(l-1)))
            print("Avg Step return: {}".format(sum(self.returns)/self.steps[-1]))

            print("========================================================\n")  



        self.episode_return = 0
        self.episode_step = 0

        self.XPos = 1.5
        self.YPos = 3
        self.ZPos = 1.5
        self.relative_pos = 0
        self.midpoint = [0,0]

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
        t = 0.1
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
        velocity_diff = self.velocity_max - self.velocity_min 
        self.velocity = self.velocity_min + velocity_diff * action[0]  
        # self.velocity = self.velocity_min
        #self.degree = round(-50 + 100 * action[1])
        original_z = self.ZPos
        original_x = self.XPos
        movements = self.movement(self.velocity, self.XPos, self.YPos, self.ZPos, self.degree)
        commands = self.perform_jump(movements)

        # for c in commands:
        #     self.agent_host.sendCommand(c)
        #     time.sleep(0.05)
        self.agent_host.sendCommand(commands[-1])
        self.episode_step += 1
        # self.total_steps += 1
        time.sleep(1)

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
        reward = 0
        score = 0

        displace_x = self.XPos - original_x
        displace_z = self.ZPos - original_z

        
        # if (displace_z > self.midpoint[0] + 1):
        #     reward += (1 - (displace_z - (self.midpoint[0] + 1))) * 1
        # elif (displace_z < self.midpoint[0] - 1):
        #     reward += (1 - ((self.midpoint[0] - 1) - displace_z)) * 1
        # else:
        #     reward += 1

        # if (displace_x > self.midpoint[1] + 1):
        #     reward += (1 - (displace_x - (self.midpoint[1] + 1))) * 1
        # elif (displace_x < self.midpoint[1] - 1):
        #     reward += (1 - ((self.midpoint[1] - 1) - displace_x)) * 1
        # else:
        #     reward += 1

        # print("dx:",displace_x)
        # print("dz:", displace_z)
        # print(self.midpoint)

        

        reward += (1.5 - abs(displace_x - self.midpoint[1])) * 2
        reward += (1.5 - abs(displace_z - self.midpoint[0])) * 2
        # print("R:",reward)
       


        for r in world_state.rewards:
            score = r.getValue()
            reward += score
        self.episode_return += reward

        
        print("Ep {} Step: {}".format(len(self.returns), self.episode_step))
        print("Velocity: {}".format(self.velocity))
        print("Step Score: {}".format(reward))
        # print("Degree: {}".format(self.degree))

        if score == -3:
            done = True
            time.sleep(1)
            

        return self.obs.flatten(), reward, done, dict()


    def get_observation(self, world_state):
        """
        Use the agent observation API to get a 5 x 10 grid around the agent. 
        The agent is in the center square facing up.
        Args
            world_state: <object> current agent world state
        Returns
            observation: <np.array>
        """
        obs = np.zeros((self.obs_size_x, self.obs_size_z))

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
                grid_trinary = []

                platform_remaining_size = 3
                i = 0
                prev_platform = True
                first_platform = False
                platform_blocks = []
                for z in range(self.obs_size_z):
                    row = 0
                    for x in range(self.obs_size_x):
                        blcok = grid[i]
                        if (prev_platform):
                            if (blcok in self.block_types or blcok == "glass" or blcok == "stone"):
                                grid_trinary.append(1)
                                row += 1
                            else:
                                grid_trinary.append(0)
                                row += 0       
                        else:              
                            if (platform_remaining_size != 0):
                                if (blcok in self.block_types):
                                    grid_trinary.append(1)
                                    if (not first_platform):
                                        first_platform = True
                                elif blcok == "glass":
                                    grid_trinary.append(2)
                                else:
                                    grid_trinary.append(0)
                                platform_blocks.append([z,x-2])
                            else:
                                grid_trinary.append(0)
                        i += 1

                    if (prev_platform and row == 0):
                        prev_platform = False
                    if (first_platform and platform_remaining_size != 0):
                        platform_remaining_size -= 1
                
                # i = 0
                # p = []
                # for x in grid_trinary:
                #     p.append(x)
                #     i += 1
                #     if (i % 5 == 0):
                #         i=0
                #         print(p)
                #         p=[]
                size = len(platform_blocks)
                self.midpoint = np.sum(platform_blocks, axis = 0)/size
                # self.relative_pos = np.sqrt((midpoint[0] - -1)**2 + (midpoint[1] - 1)**2)
              

                

                obs = np.reshape(grid_trinary, (self.obs_size_x, self.obs_size_z))
            
                

                break

        # print(obs)

        #calculate midpoint



        return obs

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

    def get_mission_xml(self):

        # Starting platform
        xml = "<DrawCuboid x1='0' x2='2' y1='2' y2='2' z1='0' z2='2' type='stone'/>"

        z = 2 + random.randint(self.gap_min, self.gap_max) + 2 # first platform 
        x = 1

        platform_index = 1
        max_platform = 50

        density = self.block_density * 2
        

        while platform_index < max_platform:
            block = random.choice(self.block_types)

            if platform_index > 10:
                density = self.block_density
            num_blocks = 0
            arr = [1, 0, -1]
            for i in range(3):
                for j in range(3):
                    if random.random() < density:
                        xml += "<DrawBlock x='{}' y='2' z='{}' type='{}'/>".format(x+arr[j], z+arr[i], block)
                        num_blocks += 1

            # Center block
            if random.random() < self.goal_block_density:
                xml += "<DrawBlock x='{}' y='2' z='{}' type='glass'/>".format(x+random.choice(arr), z+random.choice(arr))
            else:
                if num_blocks == 0:
                    xml += "<DrawBlock x='{}' y='2' z='{}' type='{}'/>".format(x+random.choice(arr), z+random.choice(arr), block)

            # else:
            z += random.randint(self.gap_min, self.gap_max) + 3
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
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-10, 10, -5, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-10, 10, -5, self.size) + \
                                xml +\
                                '''
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>James Bond</Name>
                        <AgentStart>
                            <Placement x="1.5" y="3" z="1.5" pitch="0" yaw="0"/>
                        </AgentStart>
                        <AgentHandlers>
                            <RewardForTouchingBlockType>
                                <Block type='glass' reward='10' />
                                <Block type='iron_block emerald_block gold_block lapis_block diamond_block redstone_block purpur_block' reward='5' />
                                <Block type='lava' reward='-3' behaviour='onceOnly' />
                            </RewardForTouchingBlockType>
                            <AbsoluteMovementCommands/>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromGrid>
                                <Grid name="self.floorAll">
                                    <min x="-2" y="-1" z="1"/>
                                    <max x="'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''



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
