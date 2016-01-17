# Practical Machine learning
# Deep learning - Artificial Neural Networks example
# Chapter 11

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer

network = buildNetwork(2, 5, 1, hiddenclass=SigmoidLayer)

data_set = SupervisedDataSet(2, 1)
data_set.addSample((0, 0), [0])
data_set.addSample((0, 1), [1])
data_set.addSample((1, 0), [1])
data_set.addSample((1, 1), [0])

trainer = BackpropTrainer(module=network, dataset=data_set, momentum=0.00, learningrate=0.10, weightdecay=0.0,
                          lrdecay=1.0)

error = 1
epochsToTrain = 0
while error > 0.0001:
    epochsToTrain += 1
    error = trainer.train()

results = network.activateOnDataset(data_set)
for i in range(len(results)):
    print data_set['input'][i][0], 'xor', data_set['input'][i][1], '=', int(results[i] > 0.5)
	
"""
0.0 xor 0.0 = 0
0.0 xor 1.0 = 1
1.0 xor 0.0 = 1
1.0 xor 1.0 = 0
"""

