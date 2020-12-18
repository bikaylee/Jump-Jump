---
layout: default
title:  Final
---

## Video
<iframe width="560" height="315" src=" " frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Project Summary:

## Approaches

### Environment 
Level of difficulty of maps: (all maps are randomly generated for each mission) <br>
1. No degree, complete 3x3 platform, glass always centered <br>
2. No degree, complete 3x3 platform, glass randomly at x = 0 <br>
3. Restricted degree, complete 3x3 platform, glass randomly at any block <br>
4. Restricted degree, incomplete 3x3 platform, glass may randomly be at any block <br>
5. Wider degree, incomplete 3x3 platform, glass may randomly be at any block <br>

<br>
### Reward System
Glass Block of Platform: +100 <br>
Other Block Type of Platform : < 90,  based on the relative distance to glass block <br>
Lava: > -10,  based on the relative distance to glass block 

<br>
### State Space
$$(s^3)(n-1)$$ <br>
$$s:$$ the max side size of among the platforms (s = 3 for 3x3 platforms) <br>
$$n:$$ the number of platforms 

<br>
### Observation Space 
Layers that store information about the next platform with one layer containing all available blocks and the other containing only glass blocks. Also, information of the current platforms or other platforms besides the next platform are excluded. 
<br>

First Layer: 5 x 10 of all available blocks
Second Layer: 5 x 10 of only glass block

<br>
### Action Space
#### Velocity
Since degree is taken into account and avoids our agent jumping onto the current platform again, the minimum velocity must be over a distance of 4.25m and the maximum velocity must be under a distance of 9m. The reason for disabling the agent jumping onto the same platform is to get correct and precise observation data for training.
<br>

#### Degree
In order to make this project more complex, the degree of turning can enable the agent to jump to any position of the platform. One of our environments has a restriction on degree range, the degree range is calculated based on the current position of the agent in relation to the platform. If the agent is on the right side of the platform, $$-\theta _{left}$$ is taken into account, else the agent is taking $$-\theta _{right}$$ into account Here is the equation: 
<br><br>
$$\theta _{left} = \tan^{-1}{\frac{X_{max} - X_{curr}}{Gap_{min} + 1}}$$ <br>
$$\theta _{right} = \tan^{-1}{\frac{X_{curr}}{Gap_{min} + 1}} $$


<br>
Our action space is a new implementation based on teleport to achieve the action of projectile motion of jumping. Here are the equations we used for constant gravitational acceleration: <br>


#### Projectile Motion in 3D (Jump Simulation)
$$
\begin{align}
&Horizontal\,(x), \,\,\,\,a_x = 0\,  &Vertical\,(y), \,\,\,\,a_y = -g \\ \hline
&V_x = V \cdot cos\theta\,  &V_y = V \cdot sin\theta + a_y \cdot \Delta t \\
&\Delta x = V_x \cdot \Delta t\,    &\Delta y = V_y \cdot \Delta t + \frac{1}{2} \cdot {\Delta t}^2 \\
\end{align}
$$

<br>

### Machine Learning Algorithm
PPO

## Evaluation

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
- [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [PPO algorithm from Kolby Nottingham](https://campuswire.com/c/GAD12D7F8/feed/133)
