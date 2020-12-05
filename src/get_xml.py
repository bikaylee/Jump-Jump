import random
def get_mission1_xml(self):

    # Starting platform
    xml = "<DrawCuboid x1='0' x2='2' y1='2' y2='2' z1='0' z2='2' type='stone'/>"

    z = 2 + random.randint(self.gap_min, self.gap_max) + 2 # first platform 
    x = 1

    platform_index = 1
    max_platform = 50

    density = 1
    
    while platform_index < max_platform:
        block = random.choice(self.block_types)

        num_blocks = 0
        arr = [1, 0, -1]
        for i in range(3):
            for j in range(3):
                if random.random() < density:
                    xml += "<DrawBlock x='{}' y='2' z='{}' type='{}'/>".format(x+arr[j], z+arr[i], block)
                    num_blocks += 1


        xml += "<DrawBlock x='{}' y='2' z='{}' type='glass'/>".format(x, z)

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
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-50, self.size, -50, self.size) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-50, 50, -50, 50) + \
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
                            <Block type='glass' reward="'''+str(int(self.goal_reward))+'''" behaviour='onceOnly' />
                            <Block type='iron_block emerald_block gold_block lapis_block diamond_block redstone_block purpur_block' reward='10' behaviour='onceOnly' />
                            <Block type='lava' reward="'''+str(int(self.penalty))+'''" behaviour='onceOnly' />
                        </RewardForTouchingBlockType>
                        <AbsoluteMovementCommands/>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromGrid>
                            <Grid name="self.floorAll">
                                <min x="-'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z-self.obs_size_z+1))+'''"/>
                                <max x="'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z))+'''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''

def get_mission2_xml(self):

    # Starting platform
    xml = "<DrawCuboid x1='0' x2='2' y1='2' y2='2' z1='0' z2='2' type='stone'/>"

    z = 2 + random.randint(self.gap_min, self.gap_max) + 2 # first platform 
    x = 1

    platform_index = 1
    max_platform = 50

    density = self.block_density * 2
    
    while platform_index < max_platform:
        block = random.choice(self.block_types)

        if platform_index > 20:
            density = self.block_density
        num_blocks = 0
        arr = [1, 0, -1]
        for i in range(3):
            for j in range(3):
                if random.random() < density:
                    xml += "<DrawBlock x='{}' y='2' z='{}' type='{}'/>".format(x+arr[j], z+arr[i], block)
                    num_blocks += 1

        # Center block
        if random.random() < 1:
            xml += "<DrawBlock x='{}' y='2' z='{}' type='glass'/>".format(x, z)
        else:
            if num_blocks == 0:
                xml += "<DrawBlock x='{}' y='2' z='{}' type='glass'/>".format(x, z)

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
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-50, self.size, -50, self.size) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-50, 50, -50, 50) + \
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
                            <Block type='glass' reward="'''+str(int(self.goal_reward))+'''" behaviour='onceOnly' />
                            <Block type='iron_block emerald_block gold_block lapis_block diamond_block redstone_block purpur_block' reward='10' behaviour='onceOnly' />
                            <Block type='lava' reward="'''+str(int(self.penalty))+'''" behaviour='onceOnly' />
                        </RewardForTouchingBlockType>
                        <AbsoluteMovementCommands/>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromGrid>
                            <Grid name="self.floorAll">
                                <min x="-'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z-self.obs_size_z+1))+'''"/>
                                <max x="'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z))+'''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_mission3_xml(self):

    # Starting platform
    xml = "<DrawCuboid x1='0' x2='2' y1='2' y2='2' z1='0' z2='2' type='stone'/>"

    z = 2 + random.randint(self.gap_min, self.gap_max) + 2 # first platform 
    x = 1

    platform_index = 1
    max_platform = 50

    density = self.block_density * 2
    
    while platform_index < max_platform:
        block = random.choice(self.block_types)

        if platform_index > 20:
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
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-50, self.size, -50, self.size) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='lava'/>".format(-50, 50, -50, 50) + \
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
                            <Block type='glass' reward="'''+str(int(self.goal_reward))+'''" behaviour='onceOnly' />
                            <Block type='iron_block emerald_block gold_block lapis_block diamond_block redstone_block purpur_block' reward='10' behaviour='onceOnly' />
                            <Block type='lava' reward="'''+str(int(self.penalty))+'''" behaviour='onceOnly' />
                        </RewardForTouchingBlockType>
                        <AbsoluteMovementCommands/>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromGrid>
                            <Grid name="self.floorAll">
                                <min x="-'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z-self.obs_size_z+1))+'''"/>
                                <max x="'''+str(int(self.obs_size_x/2))+'''" y="-1" z="'''+str(int(self.obs_size_z))+'''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''