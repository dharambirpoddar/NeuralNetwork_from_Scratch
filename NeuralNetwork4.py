#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:35:19 2023

Added Entropy Losses 
Lec 8 Youtube NNFS
"""


import numpy as np
import random
import nnfs 
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
#np.random.seed(0)

nnfs.init()

X,y=spiral_data(samples=100, classes=3)  # X feature , y label data or classification


class Layer_dense:
    
    def __init__(self, n_inputs,n_neurons):
        self.weights=0.1*np.random.randn (n_inputs,n_neurons) # made a random weights
        self.biases=np.zeros((1,n_neurons))   # made a biases 
        
    def forward(self, inputs):
        self.output=np.dot(inputs,self.weights)+self.biases
        
class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
        
class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        Prob_values=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=Prob_values

class Loss:
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
    
class Loss_catagoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihood=-np.log(correct_confidences)
        return negative_log_likelihood
        


layer1=Layer_dense(2, 5 )
layer2=Layer_dense(5, 3)
Activation1=Activation_Softmax()
Activation2=Activation_Softmax()

layer1.forward(X)
#print('After Activation')
Activation1.forward(layer1.output)
#layer 2
layer2.forward(Activation1.output)
Activation2.forward(layer2.output)

loss_function= Loss_catagoricalCrossentropy()
loss= loss_function.calculate(Activation2.output, y)

print("Loss",loss)



#%%
plt.polar(layer1.output,"*")
plt.show()
plt.polar(Activation1.output,"*")
plt.show()
plt.polar(layer2.output,"*")
plt.show()
plt.polar(Activation2.output,"*")
plt.show()
print(Activation1.output)
#%%
