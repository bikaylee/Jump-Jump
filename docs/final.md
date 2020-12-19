## Video

## Project Summary:


## Approaches

### Environment 
Level of difficulty: (all maps are randomly generated for each mission) <br>
1. No degree, complete 3x3 platform, glass always centered <br>
2. No degree, complete 3x3 platform, glass randomly at x = 0 <br>
3. Restricted degree, complete 3x3 platform, glass randomly at any block <br>
4. Restricted degree, incomplete 3x3 platform, glass may randomly be at any block <br>
5. Wider degree, incomplete 3x3 platform, glass may randomly be at any block <br>

<br>
### Reward System
**Glass Block of Platform:** +100 <br>
**Other Block Type of Platform:** < 90,  based on the relative distance to glass block <br>
**Lava:** > -10,  based on the relative distance to glass block 


<br>
### State Space
All possible combinations of of Observation States and Action States: <br>

**Observation State Space:** $$2^{2 \cdot 5 \cdot 10} = 2^{100}$$ <br>
**Action State Space:** $$\infty$$, infinite number of states because the variables of the action space are continuous. <br>
**State space:** $$\infty \cdot 2^{100}$$ => $$\infty$$<br>


<br>
### Observation Space 
Layers that store information about the next platform with one layer containing all available blocks and the other containing only glass blocks. Also, information of the current platforms or other platforms besides the next platform are excluded. 
<br>

**First Layer:** 5 x 10 of all available blocks <br>
**Second Layer:** 5 x 10 of only glass block 

<br>
### Action Space
<br>
#### Velocity
Since degree is taken into account and avoids our agent jumping onto the current platform again, the minimum velocity must be over a distance of 4.25m and the maximum velocity must be under a distance of 9m. The reason for disabling the agent jumping onto the same platform is to get correct and precise observation data for training. <br>
**Velocity = [8.05, 11.72]**

<br><br>
#### Degree
In order to make this project more complex, the degree of turning can enable the agent to jump to any position of the platform. One of our environments has a restriction on degree range, the degree range is calculated based on the current position of the agent in relation to the platform. If the agent is on the right side of the platform, $$-\theta _{left}$$ is taken into account, else the agent is taking $$-\theta _{right}$$ into account Here is the equation: 
<br><br>
$$\theta _{left} = \tan^{-1}{\frac{X_{max} - X_{curr}}{Gap_{min} + 1}}$$ <br>
$$\theta _{right} = \tan^{-1}{\frac{X_{curr}}{Gap_{min} + 1}} $$ <br> 

In another environment, we granted the agent a relatively more complete control of the degree. However, an increase to the range of choices, especially when it is a continuous state, can greatly impact the reinforcement learning process by adding more noise and complexity to the model. We would want to choose an optimal range of degrees that will add challenge to the agent’s learning process without making the model over complex. The max degree the agent required to travel from one platform to the next platform in the most extreme case is approximately 53 degree. In this environment, the agent is allowed to choose from a larger range of degrees [-53, 53] without other restrictions. <br>
**Degree = [-53, 53]**

<br>
##### Projectile Motion in 3D (Jump Simulation)
The results and the process of the actions are retrieved utilizing the projectile motions formulas. Previously our projectile motions calculation was limited to two dimensions (Y displacement and Z displacement). Now our projectile movement function is able to calculate the projectile motion in three dimensions (X, Y, and Z) to simulate the projectile motion under the influence of horizontal degrees. Here are the equations we used for constant gravitational acceleration: <br>

$$V\theta$$: Vertical degree
$$H\theta$$: Horizontal degree

<br>
$$
\begin{array} {ll}
&Horizontal\,(x)\, & &Frontal\,(z)\, & &Vertical\,(y)\, \\ \hline
&V_x = V cos \theta_V \cdot cos \theta_H\, & &V_z = V cos \theta_V \cdot sin \theta_H\, & &V_y = V sin \theta_V + a_y \Delta t \\
&a_x = 0\,  & &\,a_z = 0\, & &  \,a_y = -g \\
&\Delta x = V_x \Delta t\, & &\Delta z = V_z \Delta t\, &  &\Delta y = V_y \Delta t + \frac{1}{2} {\Delta t}^2 \\
\end{array}
$$


<br>

### Machine Learning Algorithm
This project includes continuous actions for both degree and velocity, so the environment is difficult and the state space is fairly large. Deep reinforcement learning of PPO is used in solving this problem. 

PPO










## Evaluation
An important aspect of your project, as I’ve mentioned several times now, is evaluating your project. Be clear and precise about describing the evaluation setup, for both quantitative and qualitative results. Present the results to convince the reader that you have solved the problem, to whatever extent you claim you have. Use plots, charts, tables, screenshots, figures, etc. as needed. I expect you will need at least a few paragraphs to describe each type of evaluation that you perform.





### Qualitative



### Quantitative
#### Environment 1
In difficulty one, the map generated for the agent would be always a 3x3 map with a center block in the middle of every platform. Also the agent's action state only allows him to jump forward in a velocity range from 8.05 to 11.72. 
At the beginning of the process, we found that the agent would tend to choose the distance close to itself to perform a jump. After 15 hours training, the ability of this agent to learn to jump further was not very outstanding; The agent would tend to jump to mainly three positions. Firstly, the position closer to himself, and then fall into the lava. Secondly, the position between glass_type and regular_block. Thirdly, the position that is very far away from himself. 

#### Environment 2
In difficulty two, the map generated for the agent would be slightly different than the map at difficulty 1. When the platform remains completed in difficulty two, the position of the glass_type_block(center block) would either be one block forward or one block behind to increase the difficulty for the agent to learn. The agent would also pick a velocity in the range from 8.05 to 11.72 with no turning degree. 
In the testing process of difficulty two, the agent would perform like what he did in the difficulty one, by choosing the maximum speed and minimum speed to perform a jump. Later on, the agent will choose to land closer to the glass_type_block. Especially, he will likely choose the position between the regular_block and the glass_type_block.

#### Environment 3

#### Environment 4

#### Environment 5





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


