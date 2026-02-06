# RLforFishingRouting
Use reinforcement learning to train agent to optimize fishing route.
The case is pretty simple, so please do not over think. The training environment is deterministic, so you basically know the optimal solution by hand calculation.
The case is to show you that a trained agent can do the same task and give you the correct answer.

# Steps to get the code running

Step 1: Install libraries from the requirements.txt by typing command pip install requirements.txt or in whatever way you like.

Step 2: Go to Training_testing.py to test the virtual environment, train an agent, test the performance of a trained agent. Animation will show up a while after clicking run to test an environment or to run some episodes to see the performance of a trained agent.


# Functions of different files
Files with names from envRL_masking_obj1 to envRL_masking_obj5 and envRL_masking_obj8 is to create a virtual fishing environment for training. Each file creates one virtual environment.

The action space, observation space, state space, reward function can be different depending on the objective that the agent trys to achieve.

File with name "main" is to define the fishing route network.

File with name NSGA is not part of the reinforcement learning. So, just leave it as it is.

File with name Env_training_testing is the file which includes all the code to set up a training environment, testing the environment, trainining of an agent, testing etc. In case you prefer to work in one script.

File with name envRL_noMask is the file without masking function. Train a proper agent will take much longer time without the masking function. I include this file in case you want to try it out.


# notebooks
The files in directory notebooks are the replicates the all the .py files. So the function descriptions above apply to the jupyter notebook files.

# Training/Model directory
The Training/Model directory contains some pretrained agents/models. In case you don't want to spend time to train agents, especially those trained with millions of steps/episodes.

# About Training logs
A Log directory will be created when you train agents. Then you can use tensorboard to see the learning process. To use tensorboard, write tensorboard --logdir Training/Logs in your IDE terminal, then you wil get a link. You will be directed to tensorboard by clicking the link.

