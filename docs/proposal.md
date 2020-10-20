---
layout: default
title:  Proposal
---

## Summary of the Project
The goal of our Ranch Crossing AI is to observe and learn about the optimal methods of breeding animals in Minecraft. The agent is given with a sword and the necessary items that are required for animal breeding, for example: carrots, potatoes and beetroots. The agent will use the items to interact with animals in the ranch and discover the proper item that can trigger the love mode of a type of animal. The use of improper items to interact with animals may lead to the decrease of population. The agent will then learn about how love mode can be utilized to breed animals with the same species. And eventually the agent will obtain the ability to rapidly increase the animal population and to simultaneously breed a large amount of animals in a given time. 

Input: the observation state(the animal population, items in the toolbar and items in the inventory).

Output: A series of optimal actions to achieve certain goals(interactions with animals).


## AI/ML Algorithms
Reinforcement Learning, Dijkstra’s Shortest Path Algorithm(Navigation), and more later. 

## Evaluation Plan
#### Quantitative evaluation: 
The baseline is that the agent will learn how to make one pair of animals to produce their offspring by feeding them the right food and not to kill animals. The main goal is to breed animals in order to reach a certain amount of offspring. Metrics include time used, a time frame, and the amount of animals it can produce. One skill that is expected for learning is the ability to pick the right food to feed the right pair of animals. The second skill is knowing the optimal way to achieve the maximum amount of offsprings possible. The optimal goal is to produce the maximum amount of offsprings within the least amount of time. 

#### Qualitative evaluation:
In order to show the project works, the agent will count the number of animals in the farm, and when the quantity of a specific type of animal has raised to a certain number that means we reach the goal. Otherwise, if the agent killed some animals by accident, the mission will be restarted.


## Appointment with the Instructor

Our meeting time is: 3:15pm - 3:30pm, Friday, October 23, 2020
