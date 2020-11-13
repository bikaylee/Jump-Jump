---
layout: default
title:  Status
---

### Video
<iframe width="560" height="315" src=" " frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Project Summary
Some minor changes have been made to the proposed ideas to enhance the creativity and interest of our project. Instead of keeping the platforms in a fixed size (3x3), we added a 1x1 size platform type. The reward of stepping on the 1x1 size platform is the same as stepping on the center block of a 3x3 size platform In addition, we successfully simulated the jumping mechanism of the original game in the minecraft world. A mission generator is implemented to randomly generate platforms for a mission. Our agent can perceive the environment through the agent states which includes the x, y, z coordinates and its velocity. Our following goal is to complete the implementation of tabular Q-learning algorithms.<br>


### Approach
Give a detailed description of your approach, in a few paragraphs. You should summarize the main algorithm you are using, such as by writing out the update equation (even if it is off-the-shelf). You Should also give details about the approach as it applies to your scenario.  For example, if you are using reinforcement learning for a given scenario, describe the setup in some detail, i.e. how many states/actions you have, what does the reward function look like. A good guideline is to incorporate sufficient details so that most of your approach is reproducible by a reader. I encourage you to use figures for this, as appropriate,as we used in the writeups for the assignments. I recommend at least 2-3 paragraphs.<br><br>

#### Environment & Reward System
Jumping platform Center: +100 <br>
Other area of platform: +10 <br>
Lava: -10 <br><br>

#### State Space
$$(s^3)(n-1)$$ <br>
$$s:$$ the max side size of among the platforms (s = 3 for 3x3 platforms) <br>
$$n:$$ the number of platforms <br><br>

#### Action Space
Our action space is a new implementation based on transpose to achieve the action of projectile motion of jumping. Here are the equations we used for constant gravitational acceleration: <br><br>

##### Classical Mechanics
$$
\begin{align}
&Horizontal\,(x), \,\,\,\,a_x = 0\,  &Vertical\,(y), \,\,\,\,a_y = -g \\ \hline
&V_x = V \cdot cos\theta\,  &V_y = V \cdot sin\theta + a_y \cdot \Delta t \\
&\Delta x = V_x \cdot \Delta t\,    &\Delta y = V_y \cdot \Delta t + \frac{1}{2} \cdot {\Delta t}^2 \\
\end{align}
$$

This is a matrix we will be storing in order to send corresponding commands to perform projectile motion. $$\begin{bmatrix} X & v_x \\ Y & v_y \\ Z & v_z \end{bmatrix}\ $$, where X, Y, Z denotes the position of the agent and their velocity according to its axis. <br>

Here is our implementation: 
```
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
```



#### Machine Learning Algorithms
The main algorithm that we used is using tabular Q-learning to train our jumping agent. According to the lecture materials, the Q-Learning Algorithm: <br><br>
$$
\begin{aligned}
&Q(S_t, A_t)\leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma\max_a Q(S_{t+1},a)- Q(s_t, A_t)]
\end{aligned}
$$

$$S_t:$$ current state <br>
$$A_t:$$ current action <br>
$$Q(S_t, A_t):$$ old values <br>
$$\alpha:$$ learning rate <br>
$$R_{t+1}:$$ reward <br>
$$\gamma:$$ discount factor <br>
$$\max_a Q(S_{t+1},a):$$ estimate of optimal future value <br>
$$R_{t+1} + \gamma\max_a Q(S_{t+1},a)- Q(s_t, A_t):$$ temporal difference <br><br>



### Evaluation

#### Quantitaive
Since the reinforcement learning algorithm is only partially deployed. Currently, we don’t have an accurate measurement of the quantitative values. We thus do not have any graphs to showcase. However, the environmental feedback such as rewards has been established. The agent receives +1 after landing on the non-center block. Landing on the center block rewards the agent with +5. And falling to lava will deduct the reward score by 10. Earning a positive score at the end of the mission signifies that the agent has either reached at least 10 blocks or has reached at least a center block without failing. In addition to gain a further comprehension of the agent’s learning process, we plan to plot its chosen velocities of each state to see how well it is learning and to prevent overfitting. The rewards are subject to be changed.<br>

#### Qualitative
As for qualitative evaluation, the simplest method to examine the overall quality of the project is to watch the agent’s movement and achievement. By watching the movements of the agent, we can make assumptions about whether the agent is choosing movements at random or it is advancing itself. By seeing how far the agent can make us an intuitive conclusion of the performance of the agent.<br><br>




### Remaining Goals and Challenges

<br><br>

### Resources Used

Malmo
- [Mission Schemas](https://github.com/microsoft/malmo/blob/master/Schemas/MissionHandlers.xsd)
- [Malmo XML Schema Documentation](https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html)

Physics
- [Classical Mechanics](https://en.wikipedia.org/wiki/Classical_mechanics) 
- [Physics: Projectile Motion](https://courses.lumenlearning.com/physics/chapter/3-4-projectile-motion/)
- [Verifying proper displacement range for particular velocity](https://www.omnicalculator.com/physics/projectile-motion)

Machine Learning Algorithm
- [Q-Learning Wiki](https://en.wikipedia.org/wiki/Q-learning)


