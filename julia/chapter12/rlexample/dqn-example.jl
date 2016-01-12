# Practical Machine learning
# Reinforcement learning - Q learning example 
# Chapter 12

using NNGraph, DeepQLearning
reload("DeepQLearning")

dqn = DeepQLearning.DQN(10,100,5)

s0 = randNNMat(10,1)
a = DeepQLearning.forward(dqn, s0)
DeepQLearning.act(dqn,s0)
DeepQLearning.learn(dqn, 0.)

s1 = randNNMat(10,1)
a = DeepQLearning.forward(dqn, s1)
DeepQLearning.act(dqn,s1)
DeepQLearning.learn(dqn, 0.)
