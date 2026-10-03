# UAV Swarm Coordination Algorithms

A UAV swarm is a group of unmanned aerial vehicles that cooperate to achieve a mission goal. Coordination algorithms decide how vehicles stay in formation, avoid collisions and divide work.

## Bio-inspired flocking

The Boids model, created by Craig Reynolds in 1986, simulates flocking with three local rules: separation, alignment and cohesion. Separation steers a vehicle away from close neighbours, alignment matches its heading to nearby vehicles, and cohesion moves it toward the local centre of mass. Because every rule uses only local information, flocking scales to large swarms without central control.

## Optimization-based methods

Particle Swarm Optimization (PSO) was introduced by Kennedy and Eberhart in 1995. Each particle adjusts its velocity using its own best known position and the best position found by the swarm. PSO is often used for path planning and parameter tuning. Ant Colony Optimization (ACO) was proposed by Marco Dorigo in the early 1990s and models how ants deposit pheromone to find short routes, which maps naturally onto route planning for multiple UAVs.

## Task allocation

The Consensus-Based Bundle Algorithm (CBBA) is a decentralized, auction-based task allocation method. Each UAV builds a bundle of tasks it bids on, and the vehicles then reach consensus on the winning bids, so no single node has to control the assignment. When a central computer is available, the Hungarian algorithm solves the optimal assignment problem in O(n^3) time, where n is the number of vehicles or tasks.

## Control architectures

In leader-follower control, one vehicle plans and the others track it. The main weakness of this design is that the leader is a single point of failure: if it is lost or jammed, the formation breaks. The virtual structure approach treats the whole formation as a rigid body and assigns each UAV a fixed slot. Fully decentralized designs avoid single points of failure at the cost of more complex coordination.
