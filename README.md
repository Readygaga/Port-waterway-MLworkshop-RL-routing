# RLforFishingRouting
Use reinforcement learning to train an agent to optimize the fishing route.
The case is pretty simple, so please do not overthink. The training environment is deterministic, so you basically know the optimal solution by hand calculation.
The case is to show you that a trained agent can do the same task and give you the correct answer.

# Steps to get the code running

Step 1: Install packages from the requirements.txt by typing the command pip install requirements.txt or in whatever way you like.

Step 2: Go to Training_testing.py to test the virtual environment, train an agent, and test the performance of a trained agent. Animation will show up a while after clicking run to test an environment or to run some episodes to see the performance of a trained agent.


# Functions of different files
Files with names from envRL_masking_obj1 to envRL_masking_obj5 and envRL_masking_obj8 are to create a virtual fishing environment for training. Each file creates one virtual environment.

The action space, observation space, state space, and reward function can be different depending on the objective that the agent tries to achieve. Therefore I created several training environment files.

The file with the name "main" is to define the fishing route network.

The file with the name NSGA is not part of the reinforcement learning. So, just leave it as it is.

The file with name Env_training_testing is the file that includes all the code to set up a training environment, testing the environment, training of an agent, and testing etc. In case you prefer to work in one script.

The file with the name envRL_noMask is the file without the masking function. Training a proper agent will take much longer time without the masking function. I include this file in case you want to try it out.


# notebooks
The files in the directory notebooks are the replicates of all the .py files. So the function descriptions above apply to the Jupyter notebook files.

# Training/Model directory
The Training/Model directory contains some pretrained agents/models. In case you don't want to spend time to train agents, especially to train them with millions of steps/episodes.

# About Training Logs
A Log directory will be created when you train agents. Then you can use TensorBoard to see the learning process. To use TensorBoard, write tensorboard --logdir Training/Logs in your IDE terminal, then you will get a link. You will be directed to TensorBoard by clicking the link.

