from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import random
import numpy as np
from numpy.random import randint



# Hyperparameters
FLOOR = 3

# map size
SIZEx = 500
SIZEz = 100

GAP = 2             
BLOCK_SMALL = 0.3   # probability of small blocks
BLOCK_SIZE = [0, 1] # 0=1x1, 1=3x3

VELOCITY_MIN = 8.06  # range = 4m 7.81
VELOCITY_MAX = 8.93 # range = 5m 8.72

PRIME_NUMBER = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
BLOCK_TYPES = ['iron_block', 'emerald_block', 'gold_block', 'lapis_block', 'diamond_block', 'redstone_block', 'purpur_block']

def GetMissionXML():

    xml = ""
    x = 4 # -SIZEz + 1
    primeInd = 0
    i = 1
    while i < 50:
        if PRIME_NUMBER[primeInd] == i:
            primeInd += 1
            l = BLOCK_SIZE[0]
            xml += "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='0' z2='0' type='glass'/>".format(x, x)
            x += GAP + 1
        else:
            l = BLOCK_SIZE[1] 
            block_type = random.choice(BLOCK_TYPES)
            xml += "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='{}'/>".format(x,x+l+1,-l,l,block_type)
            xml += "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='0' z2='0' type='glass'/>".format(x+l, x+l)
            x += GAP + 3
            if PRIME_NUMBER[primeInd+1] == i:
                x += 1
        i += 1

         

    xml += "<DrawCuboid x1='-1' x2='1' y1='2' y2='2' z1='-1' z2='1' type='stone'/>"

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
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-SIZEx, SIZEx, -SIZEz, SIZEz) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-SIZEx, SIZEx, -SIZEz, SIZEz) + \
                            xml +\
                            '''
                        </DrawingDecorator>
                        <ServerQuitFromTimeUp timeLimitMs="500000"/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>James Bond</Name>
                    <AgentStart>
                        <Placement x="0.5" y="3" z="0.5" pitch="0"/>
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
                        <VideoProducer viewpoint='1'/>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''

def movement (v, x ,y):
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


def perform_jump(movementPath):
    path = []
    for a in movementPath:
        x = a[0]
        y = a[1]
        path.append("tp {} {} 0.5".format(round(x,4),round(y,4)))
    return path

 
# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 10

for i in range(num_repeats):
    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    # Attempt to start a mission:
    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running ", end=' ')

    
    
    action_list = []

    movement_path = movement(VELOCITY_MAX, 0.5, 3)
    action_list.append(perform_jump(movement_path))

    num_jump = 0
    prime_index = 0
    #x = movement_path[-1][0]
    x = 0.5 + GAP + 3

    while num_jump < 20:
        num_jump += 1
        
        #if block is small, next must be large
        if PRIME_NUMBER[prime_index] == num_jump: 
            prime_index += 1
            movement_path = movement(VELOCITY_MIN, x, 3)
            #x = movement_path[-1][0]
            x += GAP + 2
            action_list.append(perform_jump(movement_path))

        #if block is large
        else:

            #if next block is small
            if PRIME_NUMBER[prime_index] == num_jump+1: 
                movement_path = movement(VELOCITY_MIN, x, 3)
                x += GAP + 2
            #if next block is large
            else:
                movement_path = movement(VELOCITY_MAX, x, 3)
                x += GAP + 3

            action_list.append(perform_jump(movement_path))
            #x = movement_path[-1][0]



    # Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.1)
        
        time.sleep(3)
        #Sending the next commend from the action list -- found using the Dijkstra algo.
        for a in action_list:
            time.sleep(1.5)
            print()
            for index in range(len(a)):
                #agent_host.sendCommand(a[index])
                time.sleep(0.02)
                #print(a[index])
            

        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission ended")
    # Mission has ended.

