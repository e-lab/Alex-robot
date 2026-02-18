# Alex humanoid robot capabilities planning

Developing a plan for the design of software and control for a humanoid robot.

![](images/alex.png)

## Platform
- IHMC Alex https://www.ihmc.us/news20251119/

- Alex onboard compute capabilities: NUC, NVIDIA Jetson AGX, ZED X mini in head and belly, GPS.

## Targets

Building exploration, entering a building, navigating up and downstairs, force doors open, open cabinets, explore occluded spaces, find rooms occupied by people and report how many people / threats

This a list of target capabilities for IHMC Alex:

### Overall target capabilities
- walk,  soft falling, standing from ground, open doors with different hinges, donkey kick door, front kick door, use a battering ram or tool

### Perception
- localize against a map, semantic object recognition and localization, object pose, scene graph, identify occlusions, identify people, infer enemies and dangerous objects.

### Navigation
- navigate to a place on map / world map from GPS coordinate, 

### Manipulation:
- Grasp novel objects from point cloud, Dextrous manipulation, Open cabinets, drawers, Move furniture

### Global planning:

- Remember location. Of key objects, query scene graph, determine sequence of tasks, decompose actions, determine where to go to a full map, determine when to search occluded areas and cabinets

### User interfaces:
- Text to commands, speech to text to commands


## Plans for simulations

Here is a list of plans to develop a complete simulation environment for Alex and target capabilities:

### Simulator
- IsaacSim https://github.com/isaac-sim/IsaacSim high-fidelity, 
- MuJoCo https://mujoco.readthedocs.io/en/stable/overview.html has UniTree h1 robot, is lightweight
- IHMC https://github.com/ihmcrobotics

### Exploration
- Multi-room environment with functional doors, some cabinets, can position various objects
- https://allenai.org/blog/molmospaces


## Planning
- Receive instruction from operator: voice or text
- Plan activities using large LLM, divide task into individual action sets, define completion for each action and overall task
- Loop over planned activities


## Moving


## Manipulating


 
