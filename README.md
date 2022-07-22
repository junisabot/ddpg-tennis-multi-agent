[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# MADDPG - Tennis Multi Agent

### Introduction

To train a pair of reinforcement learning agent to play table tennis.

![Trained Agent][image1]

### Project Details

- This environment is simulated using Unity's Reacher Environment
- The observation space consists of 8 variables corresponding to position, velocity of the ball, and racket.
- The action space consist of 2 variables corrensponding to left right movement of racket, and jumping.
- A position reward, +0.1 is given if any agent hits the ball over the net.
- A negative reward, -0.01 is given if any agent hits the ball out of bound or the ball hit the ground.
- The goal of this project is to keep the ball in play.
- In order to solve the environment, your agent must get an average score of +0.5 over 100 consecuitive episodes.

### Getting Started

1. Clone this repo:
```shell
$ git clone https://github.com/junisabot/ddpg-tennis-multi-agent.git
```

2. Install python dependencies through pip:
```shell
$ pip install -r requirement.txt
```

3. Download Unity environment according your OS to the root folder of this repo and unzip the file.
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### Project Structure

1. report.html is a report of this project
2. train.ipynb is the notebook to train MADDPG network with this project
3. two agents modules located in directory modules/, named actorA (left player) & actorB (right player).
4. network/actor.py contains actor neural network from DDPG.
5. network/critic.py contains critic neural network from DDPG.
6. config.py contains all the adjustable parameters of this project.
7. pretrained models are provided in directory ./pretrained_model
