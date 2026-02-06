# RLforFishingRouting
Use reinforcement learning to train agent to optimize fishing route

Step 1: Install libraries from the requirements.txt

Step 2: Go to Training_testing.py to test the virtual environment, train an agent, test the performance of a trained agent. Animation will show up a while after clicking run.


# Functions of different files
Files with names from envRL_masking_obj1 to envRL_masking_obj5 and envRL_masking_obj8 is to create a virtual fishing environment for training. Each file creates one virtual environment.

The action space, observation space, state space, reward function can be different depending on the objective that the agent trys to achieve.

File with name "main" is to define the fishing route network.

File with name NSGA is not part of the reinforcement learning. So, just leave it as it is.

File with name Env_training_testing is the file which includes all the code to set up a training environment, testing the environment, trainining of an agent, testing etc. in case you prefer to work in one script.

File with name envRL_noMask is the file without masking function. Train a proper agent will take much longer time without the masking function. I include this file in case you want to try it out.
