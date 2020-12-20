CS175 Fall 2020

# Pixel Jump

Our AI project is a Jump & Jump gameplay simulation in Minecraft, with different sizes and shapes of platform for the agent to jump on based on gravity and velocity. 

---
### Website
https://bikaylee.github.io/Pixel-Jump/.

### Basics 
- [x] Classical Mechanics
- [x] Deep Reinforcement Learning
- [x] Proximal Policy Optimization

#### Level of Difficulty 
**all maps are randomly generated for each mission** <br>
1. No degree, complete 3x3 platform, glass always centered <br>
2. No degree, complete 3x3 platform, glass randomly at x = 1.5 <br>
3. Restricted degree, complete 3x3 platform, glass randomly at any block <br>
4. Restricted degree, incomplete 3x3 platform, glass may randomly be at any block <br>
5. Wider degree, incomplete 3x3 platform, glass may randomly be at any block <br>

#### Reward System
**Glass Block of Platform:** +100 <br>
**Other Block Type of Platform:** < 90,  based on the relative distance to glass block <br>
**Lava:** < -10,  based on the relative distance to glass block 

#### Observation Space 
Layers that store information about the next platform with one layer containing all available blocks and the other containing only glass blocks. Also, information of the current platforms or other platforms besides the next platform are excluded. 
<br>

**First Layer:** 5 x 10 storing all available blocks information and air <br>
**Second Layer:** 5 x 10 storing only glass block information and air <br>


### Game Demo GIF
<img src="http://g.recordit.co/3kgNawkGHM.gif" width=250><br>

## References
Malmo
- [Mission Schemas](https://github.com/microsoft/malmo/blob/master/Schemas/MissionHandlers.xsd)
- [Malmo XML Schema Documentation](https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html)

Physics
- [Physics: Projectile Motion in 2D](https://courses.lumenlearning.com/physics/chapter/3-4-projectile-motion/)
- [Physics: Projectile Motion in 3D](https://people.physics.tamu.edu/mahapatra/teaching/ch3.pdf)
- [Verifying proper displacement range for particular velocity](https://www.omnicalculator.com/physics/projectile-motion)

Machine Learning Algorithm
- [RLlib](https://docs.ray.io/en/latest/rllib.html)
- [RLlib Pytorch Models](https://docs.ray.io/en/latest/rllib-models.html#pytorch-models)
- [PPO OpenAI](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [PPO algorithm from Kolby Nottingham](https://campuswire.com/c/GAD12D7F8/feed/133)

Tools 
- [Draw.io](http://draw.io/)
- [Jupyter Notebook](https://docs.anaconda.com/ae-notebooks/user-guide/basic-tasks/apps/jupyter/)
- [LaTex Overleaf](https://www.overleaf.com/learn)
- [Table Generator](https://www.tablesgenerator.com/latex_tables)
- [Linear Regression](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html)

