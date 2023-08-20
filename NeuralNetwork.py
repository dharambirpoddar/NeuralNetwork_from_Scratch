#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:06:26 2023

@author: ubuntu
Neural Network Code from Scratch Help:https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3
"""
#%%
x=[1,2,3,2.5]  # input layer
weight_1=[0.2,0.8,-0.5,1]
weight_2=[0.5,-0.91,0.26,-0.5]
weight_3=[-0.26,-0.27,0.17,0.87]
b1=2 
b2=3 
b3=0.5


y=[weight_1[0]*x[0]+weight_1[1]*x[1]+weight_1[2]*x[2]+b1,
   weight_2[0]*x[0]+weight_2[1]*x[1]+weight_2[2]*x[2]+b1,
   weight_3[0]*x[0]+weight_3[1]*x[1]+weight_3[2]*x[2]+b1 ]

print(y)
#%%
input_layer=x
weights=[weight_1,weight_2,weight_3]
biases=[b1,b2,b3]


output_neuron=[]
for neuron_w,neuron_b in zip(weights,biases): # value taken from both array element wise simultaneously
    print(neuron_w, neuron_b)
    new_neuron=0
    for neu_w, input_neu in zip(neuron_w,input_layer):
        
        new_neuron+=neu_w*input_neu
    new_neuron+=neuron_b
    
    output_neuron.append(new_neuron)
print(output_neuron)


#%% Using Numpy
import numpy as np

input_layer=x
weights=[weight_1,weight_2,weight_3]
biases=[b1,b2,b3]

output_layer=np.dot(weights,input_layer)+biases




         













