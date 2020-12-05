CS175 Fall 2020

# Pixel Jump

Our AI project is a Jump & Jump gameplay simulation in Minecraft, with different sizes and shapes of platform for the agent to jump on based on gravity and velocity. 

---
### Website
https://bikaylee.github.io/Pixel-Jump/.

### Basics 
- [x] Classical Mechanics
- [ ] Reinforcement Learning
- [ ] Proximal Policy Optimization

#### Level of Difficulty 
1: "Easy"   3x3 platform consists of 9 blocks with goal block centered and without turning degree <br>
2: "Medium" 3x3 platform consists of 9 blocks with goal randomly shown on z-axis and without turning degree <br>
3: "Hard"   3x3 platfrom consists of 1 to 9 blocks with goal block randomly shown at any blocks and with turning degree

#### Observation 
Agent will be able to observe all block types on the next platorm by two return lists of 0s and 1s.  

#### Reward System
The more closer the agent gets to the goal block, more points will be rewarded according to its relative position. 

### Game Demo GIF
<img src="http://g.recordit.co/3kgNawkGHM.gif" width=250><br>

### Reference 
Malmo
- https://github.com/microsoft/malmo/blob/master/Schemas/MissionHandlers.xsd
- https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html

Physics
- https://en.wikipedia.org/wiki/Classical_mechanics
- https://courses.lumenlearning.com/physics/chapter/3-4-projectile-motion/
- https://www.omnicalculator.com/physics/projectile-motion
- https://people.physics.tamu.edu/mahapatra/teaching/ch3.pdf

Machine Learning Algorithm
- https://en.wikipedia.org/wiki/Q-learning
- https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c
- https://spinningup.openai.com/en/latest/algorithms/ppo.html

