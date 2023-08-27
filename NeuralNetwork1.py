#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 01:12:24 2023

@author: ubuntu
"""
import numpy as np

input_layer=[[1,2,3,2.5],  # multiple sample here we are using 3 sample, usually 32 sample is used in 1 batch      
             [2,5,-1,2],
             [-1.5,2.7,3.3,-0.8]]

weights=[[0.2, 0.8, -0.5, 1], 
         [0.5, -0.91, 0.26, -0.5], 
         [-0.26, -0.27, 0.17, 0.87]]

biases=[2,3,0.5]

weights_2=[[0.1,-0.14,0.5],
           [-0.5,0.12,0.33],
           [-0.44,0.73,-0.13]]
biases_2=[-1,2,-0.5]

output_layer1=np.dot(input_layer,np.array(weights).T)+biases
output_layer2=np.dot(output_layer1,np.array(weights_2).T)+biases_2

print(output_layer2)

#%%
import numpy as np
import random

#np.random.seed(0)

X=[[1,2,3,2.5],  # multiple sample here we are using 3 sample, usually 32 sample is used in 1 batch      
   [2,5,-1,2],
   [-1.5,2.7,3.3,-0.8]]

class Layer_dense:
    
    def __init__(self, n_inputs,n_neurons):
        self.weights=np.random.randn (n_inputs,n_neurons) # made a random weights
        self.biases=np.zeros((1,n_neurons))   # made a biases 
        
    def forward(self, inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

layer1=Layer_dense(4, 3)
layer2=Layer_dense(3,2)

output1=layer1.forward(X)
print(output1)
output2=layer2.forward(output1)
print(output2)




















