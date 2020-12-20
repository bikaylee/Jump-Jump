---
layout: default
title:  Home
---


<img src="image/MalmoDemo.gif" width=600><br>

Pixel Jump is a game performing physical jump from one point to another with user-control gravitational force and velocity and horizontal degree. The purpose of this project is to let the agent observe his surrounding platforms and then pick an initial velocity from a continuous action space to ensure that the agent can land on various positions of the next platform and hopefully jump onto the glass (goal) block for greater reward. 

Source Code:  [https://github.com/bikaylee/Pixel-Jump]( https://github.com/bikaylee/Pixel-Jump)


### Environment 
Level of difficulty: (all maps are randomly generated for each mission) <br>
1. No degree, complete 3x3 platform, glass always centered <br>
2. No degree, complete 3x3 platform, glass randomly at x = 1.5 <br>
3. Restricted degree, complete 3x3 platform, glass randomly at any block <br>
4. Restricted degree, incomplete 3x3 platform, glass may randomly be at any block <br>
5. Wider degree, incomplete 3x3 platform, glass may randomly be at any block <br>

<br>
### Reward System
**Glass Block of Platform:** +100 <br>
**Other Block Type of Platform:** < 90,  based on the relative distance to glass block <br>
**Lava:** < -10,  based on the relative distance to glass block 



Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)
